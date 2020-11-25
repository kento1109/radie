import re
from typing import List, Union, Tuple
from itertools import chain, groupby

from pydantic import BaseModel
import MeCab

from radie.src.utils.ner_utils import Tagger
from radie.src.utils.re_utils import Rex
from radie.src.utils.sc_utils import SC
from radie.src.utils import candidate_generation
from radie.src.utils.candidate_generation import Entity
from radie.src.utils import preprocessing

SENT_HEAD_MAP = {'head': 0, 'not_head': 1}


class OAModel(BaseModel):
    """構造化結果をObject-Attributeの形式を保持するクラス"""
    findings_seq: int
    obj_entity: Entity
    attr_entity: Entity
    relation_score: float

    def chunking(self):
        self.obj_entity.chunking()
        self.attr_entity.chunking()


class OAVTripletModel(BaseModel):
    """構造化結果をOAVTripletの形式で保持するクラス"""
    findings_seq: int
    obj_tokens: List[str]
    attr_tokens: List[str]
    value_entity: str
    relation_score: float

    def chunking(self):
        self.obj_tokens = ''.join(self.obj_tokens)
        self.attr_tokens = ''.join(self.attr_tokens)


class Structured_Report(BaseModel):
    structured_data_list: Union[List[OAModel], List[OAVTripletModel]]

    def __iter__(self):
        return iter(self.structured_data_list)

    def __getitem__(self, i):
        return self.structured_data_list[i]

    def __len__(self):
        return len(self.structured_data_list)

    def chunking(self):
        if self.structured_data_list:
            for structured_data in self.structured_data_list:
                structured_data.chunking()


# class OAVTripletWrapper(object):
#     def __init__(self, ner_path, re_path):
#         self.tagger = Tagger(ner_path)
#         self.rex = Rex(re_path)
#         self.cg = candidate_generation
#         self.cg.set_marker_info()

#     def get_oav_triplet(self, tokens, do_certainty_completion=False):
#         """
#         do_certainty_completion: certaintyが存在しない場合、暗黙的肯定を補完するか
#         """
#         labels = self.tagger.predict(tokens)
#         entities = self.cg.get_entities(labels)
#         obj_entities = list(
#             filter(lambda entity: entity[0] in self.cg.OBJ_NAMES, entities))
#         relatin_statements = self.cg.create_relation_statements(tokens, labels)
#         oav_list = []
#         if relatin_statements:
#             # get relation scores
#             relation_scores = self.rex.predict(relatin_statements)
#             for statement, score in zip(relatin_statements, relation_scores):
#                 oav_list.append(
#                     OAVTripletModel(obj_tokens=statement.obj.tokens,
#                                     attr_tokens=statement.attr.tokens,
#                                     value_entity=statement.attr.name,
#                                     relation_score=score))
#             if do_certainty_completion:
#                 oav_list += certainty_completion(tokens, obj_entities)
#         else:
#             if do_certainty_completion:
#                 if obj_entities:
#                     oav_list += certainty_completion(tokens, obj_entities)
#         return oav_list


class Structured_Reporting(object):
    def __init__(self,
                 ner_path,
                 re_path=None,
                 sc_path=None,
                 do_preprocessing=True,
                 do_split_sentence=True,
                 do_tokenize=True,
                 return_as_oav_format=False,
                 do_certainty_completion=False):

        self.tagger = Tagger(ner_path)
        if re_path is not None:
            self.rex = Rex(re_path)
        else:
            self.rex = None
        if sc_path is not None:
            self.sc = SC(sc_path)
        else:
            self.sc = None
        self.cg = candidate_generation
        self.cg.set_marker_info()
        self.do_preprocessing = do_preprocessing
        self.do_split_sentence = do_split_sentence
        self.do_tokenize = do_tokenize
        self.return_as_oav_format = return_as_oav_format
        self.do_certainty_completion = do_certainty_completion
        if self.do_tokenize:
            self.mc = MeCab.Tagger("-Owakati")

    def _split_sent(self, report: str) -> List[str]:
        sent_list = []
        for sent in re.split('\[SEP\]', report):
            if sent:
                sent_list.append(sent)
        return sent_list

    def _preprocessing(self, report: str) -> str:
        report = preprocessing.zen_to_han(report)
        report = preprocessing.remove_header(report)
        report = preprocessing.remove_footer(report)
        report = preprocessing.remove_char(report)
        report = preprocessing.mask_date(report)
        report = preprocessing.replace_space(report)
        report = preprocessing.insert_sep(report)
        report = preprocessing.han_to_zen(report)
        return report

    def _create_group_idx_list(self, heads_list: List[int]) -> List[int]:
        group_list = list()
        idx = -1
        for head in heads_list:
            if head == SENT_HEAD_MAP['head']:
                idx += 1
                group_list.append(idx)
            else:
                group_list.append(idx)
        return group_list

    def _group_sentence(
            self, tokens_list: List[str], group_list: List[int],
            labels_list: List[str]) -> Tuple[List[List[str]], List[List[str]]]:
        grouped_tokens_list = list()
        grouped_labels_list = list()
        groups = map(
            lambda xx: list(xx[1]),
            groupby(zip(tokens_list, labels_list, group_list),
                    key=lambda x: x[2]))
        for group in groups:
            _tokens_list = map(lambda g: g[0], group)
            _tokens_list = list(chain.from_iterable(_tokens_list))
            _labels_list = map(lambda g: g[1], group)
            _labels_list = list(chain.from_iterable(_labels_list))
            grouped_tokens_list.append(_tokens_list)
            grouped_labels_list.append(_labels_list)
        return grouped_tokens_list, grouped_labels_list

    def _certainty_completion(self,
                              tokens,
                              obj_entities,
                              findings_seq,
                              score=0.5):
        """
        score: completion時に設定するスコア
        """
        _structured_list = list()
        for obj_entity in obj_entities:
            obj_tokens = tokens[obj_entity[1]:obj_entity[2] + 1]
            if self.return_as_oav_format:
                _structured_list.append(
                    OAVTripletModel(findings_seq=findings_seq,
                                    obj_tokens=obj_tokens,
                                    attr_tokens=['implicit_positive'],
                                    value_entity='Certainty_descriptor',
                                    relation_score=score))
            else:
                _structured_list.append(
                    OAModel(findings_seq=findings_seq,
                            obj_entity=Entity(name=obj_entity[0],
                                              tokens=obj_tokens),
                            attr_entity=Entity(name='Certainty_descriptor',
                                               tokens=['implicit_positive']),
                            relation_score=score))
        return _structured_list

    def _create_structured_data(self, tokens, labels, findings_seq):
        """
        NERの結果からobject-attributeのペアを作成し、構造化形式で返す
        return_as_oav_format: 構造化形式の形式（OAVTripletのフォーマットで返すか）
        do_certainty_completion: certaintyが存在しない場合、暗黙的肯定を補完するか
        """
        entities = self.cg.get_entities(labels)
        obj_entities = list(
            filter(lambda entity: entity[0] in self.cg.OBJ_NAMES, entities))
        relatin_statements = self.cg.create_relation_statements(tokens, labels)
        structured_data_list = []
        if relatin_statements:
            # get relation scores
            relation_scores = self.rex.predict(relatin_statements)
            for statement, score in zip(relatin_statements, relation_scores):
                if self.return_as_oav_format:
                    structured_data_list.append(
                        OAVTripletModel(findings_seq=findings_seq,
                                        obj_tokens=statement.obj.tokens,
                                        attr_tokens=statement.attr.tokens,
                                        value_entity=statement.attr.name,
                                        relation_score=score))
                else:
                    structured_data_list.append(
                        OAModel(findings_seq=findings_seq,
                                obj_entity=statement.obj,
                                attr_entity=statement.attr,
                                relation_score=score))
            if self.do_certainty_completion:
                structured_data_list += self._certainty_completion(
                    tokens, obj_entities, findings_seq)
        else:
            if self.do_certainty_completion:
                if obj_entities:
                    structured_data_list += self._certainty_completion(
                        tokens, obj_entities, findings_seq)
        return structured_data_list

    def ner(self, report: str) -> Tuple[List[List[str]], List[List[str]]]:
        if self.do_preprocessing:
            report = self._preprocessing(report)
        if self.do_split_sentence:
            sent_list = self._split_sent(report)
        else:
            sent_list = [report]
        tokens_list, labels_list, heads_list = list(), list(), list()
        for sent in sent_list:
            if self.do_tokenize:
                tokens = self.mc.parse(sent).strip().split(' ')
            else:
                tokens = sent
            labels = self.tagger.predict(tokens)
            if self.sc is not None:
                is_head = self.sc.predict(tokens)
                heads_list.append(is_head)
            tokens_list.append(tokens)
            labels_list.append(labels)
        # 文を結合する
        if self.sc is not None:
            if tokens_list:
                heads_list[0] = SENT_HEAD_MAP['head']  # 先頭は0（先頭ラベル）で固定する
                self._insert_sep(heads_list, tokens_list, labels_list)
                group_list = self._create_group_idx_list(heads_list)
                tokens_list, labels_list = self._group_sentence(
                    tokens_list, group_list, labels_list)
        return tokens_list, labels_list

    def _insert_sep(self, heads_list: List[int], tokens_list: List[List[str]],
                    labels_list: List[List[str]]) -> None:
        """ 文中の区切りに[SEP]を追加する """                 
        for i, head in enumerate(heads_list):
            if head == SENT_HEAD_MAP['not_head']:
                tokens_list[i].insert(0, 'SEP')  # 文間にSEP記述子を追加
                labels_list[i].insert(0, 'O')  # SEPにはOラベルを付与する

    def structuring(
            self, report: str) -> Union[List[OAModel], List[OAVTripletModel]]:
        if self.do_preprocessing:
            report = self._preprocessing(report)
        if self.do_split_sentence:
            sent_list = self._split_sent(report)
        else:
            sent_list = [report]
        tokens_list, labels_list, heads_list = list(), list(), list()
        for sent in sent_list:
            if self.do_tokenize:
                tokens = self.mc.parse(sent).strip().split(' ')
            else:
                tokens = sent
            labels = self.tagger.predict(tokens)
            if self.sc is not None:
                is_head = self.sc.predict(tokens)
                heads_list.append(is_head)
            tokens_list.append(tokens)
            labels_list.append(labels)
        # 文を結合する
        if self.sc is not None:
            heads_list[0] = SENT_HEAD_MAP['head']  # 先頭は0（先頭ラベル）で固定する
            self._insert_sep(heads_list, tokens_list, labels_list)
            group_list = self._create_group_idx_list(heads_list)
            tokens_list, labels_list = self._group_sentence(
                tokens_list, group_list, labels_list)
        structured_data_list = list()
        for i, (tokens, labels) in enumerate(zip(tokens_list, labels_list)):
            _structured_data_list = self._create_structured_data(
                tokens, labels, i)
            structured_data_list.extend(_structured_data_list)
        return Structured_Report(structured_data_list=structured_data_list)