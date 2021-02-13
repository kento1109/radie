import os
import re
from typing import List, Union, Tuple
from itertools import chain, groupby

import MeCab
import torch
from pydantic import BaseModel
from logzero import logger
from omegaconf import OmegaConf

# from radie.src.normalizer import ChangeNormalizer
from radie.src.utils.ner_utils import Tagger
from radie.src.utils.rc_utils import RelationClassifier
from radie.src.utils.ced_utils import CertaintyClassifier
from radie.src.utils import candidate_generation
from radie.src.utils import preprocessing
from radie.src.utils import types


class Extractor(object):
    def __init__(self,
                 do_preprocessing=True,
                 do_split_sentence=True,
                 do_tokenize=True,
                 return_as_oav_format=False,
                 do_normalize=False,
                 do_certainty_completion=False,
                 do_insert_sep=False,
                 sep_token='。'):

        # load config
        radie_dir = os.path.dirname(__file__)
        self.config = OmegaConf.load(os.path.join(radie_dir, 'config.yaml'))

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.tagger = Tagger(self.config.path.model_ner)
        logger.info(f"tagger loaded ...")

        self.rc = RelationClassifier(self.config.path.model_relation_classifier)
        logger.info(f"relation classifier model loaded ...")

        # self.change_normalizer = ChangeNormalizer(self.config.path.model_change_normalizer)
        # logger.info(f"change normalizer model loaded ...")

        self.cc = CertaintyClassifier(
            self.config.path.model_certainty_classifier)
        logger.info(f"certainty classifier model loaded ...")

        self.cg = candidate_generation
        self.cg.set_marker_info()
        self.do_preprocessing = do_preprocessing
        self.do_split_sentence = do_split_sentence
        self.do_tokenize = do_tokenize
        self.return_as_oav_format = return_as_oav_format
        self.do_normalize = do_normalize
        self.do_certainty_completion = do_certainty_completion
        if self.do_tokenize:
            self.mc = MeCab.Tagger(
                f"-Owakati -d {self.config.path.mecab_dict}")
        self.do_insert_sep = do_insert_sep
        self.sep_token = sep_token

    def __call__(self, report: str) -> List[types.Concept]:
        tagger_result = self.ner(report)
        object_statements = self.cg.create_entity_statements(tagger_result)
        certainty_scales = [
            self.cc.predict(statement.tokens)
            for statement in object_statements
        ]
        return [
            types.Concept(clinical_concept="".join(
                object_statement.obj.tokens),
                          certainty_scale=certainty_scale)
            for object_statement, certainty_scale in zip(
                object_statements, certainty_scales)
        ]

    def _split_sent(self, report: str) -> List[str]:
        sent_list = []
        for sent in re.split(self.sep_token, report):
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
        report = preprocessing.insert_sep(report, self.sep_token)
        report = preprocessing.han_to_zen(report)
        return report

    def _create_group_idx_list(self, heads_list: List[int]) -> List[int]:
        group_list = list()
        idx = -1
        for head in heads_list:
            if head == 'not_related':
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

    # def _certainty_completion(self,
    #                           tokens,
    #                           obj_entities,
    #                           findings_seq,
    #                           score=0.5):
    #     """
    #     score: completion時に設定するスコア
    #     """
    #     _structured_list = list()
    #     for obj_entity in obj_entities:
    #         _start_idx, _end_idx = obj_entity[1], obj_entity[2]
    #         obj_tokens = tokens[_start_idx:_end_idx + 1]
    #         if self.return_as_oav_format:
    #             _structured_list.append(
    #                 types.OAVTripletModel(findings_seq=findings_seq,
    #                                 obj_tokens=obj_tokens,
    #                                 attr_tokens=['implicit_positive'],
    #                                 value_entity='Certainty_descriptor',
    #                                 relation_score=score))
    #         else:
    #             _structured_list.append(
    #                 types.OAModel(findings_seq=findings_seq,
    #                         obj_entity=types.Entity(name=obj_entity[0],
    #                                           tokens=obj_tokens,
    #                                           start_idx=_start_idx),
    #                         attr_entity==types.Entity(name='Certainty_descriptor',
    #                                            tokens=['implicit_positive'],
    #                                            start_idx=-1),
    #                         relation_score=score))
    #     return _structured_list

    def _create_structured_data(self, tagger_result: types.Tagger, findings_seq: int):
        """
        NERの結果からobject-attributeのペアを作成し、構造化形式で返す
        return_as_oav_format: 構造化形式の形式（OAVTripletのフォーマットで返すか）
        do_certainty_completion: certaintyが存在しない場合、暗黙的肯定を補完するか
        """
        entities = self.cg.get_entities(tagger_result.labels)
        obj_entities = list(
            filter(lambda entity: entity[0] in self.cg.OBJ_NAMES, entities))
        relatin_statements = self.cg.create_relation_statements(tagger_result)
        # change_statements = self.cg.create_entity_statements(
        #     tokens, labels,
        #     [self.cg.CHANGE_NAME]) if self.do_normalize else list()
        # norm_map_list = list()
        # for change_statement in change_statements:
        #     _norm_map = types.NormMap(start_idx=change_statement.obj.start_idx,
        #                         normed=self.change_norm.normalize(
        #                             change_statement.tokens))
        #     norm_map_list.append(_norm_map)
        structured_data_list = []
        if relatin_statements:
            # get relation scores
            for relation_statement in relatin_statements:
            # relation_scores = self.rc.predict(relatin_statements.tokens)
                print(self.rc.predict(relation_statement.tokens))
        #     for _statement, score in zip(relatin_statements, relation_scores):
        #         # do normalize
        #         # HACK: should be modified !!
        #         if self.do_normalize:
        #             # is attribute entity a change entity ?
        #             norm_map = list(
        #                 filter(
        #                     lambda _norm: _norm.start_idx == _statement.attr.
        #                     start_idx, norm_map_list)
        #             )[0] if _statement.attr.name == self.cg.CHANGE_NAME else None
        #             if norm_map:
        #                 norm_list = list()
        #                 if isinstance(norm_map.normed, list):
        #                     for _norm in norm_map.normed:
        #                         norm_list.append(_norm)
        #                     _statement.attr.norms = norm_list
        #                 else:
        #                     _statement.attr.norms = norm_map.normed
        #         if self.return_as_oav_format:
        #             structured_data_list.append(
        #                 types.OAVTripletModel(findings_seq=findings_seq,
        #                                 obj_tokens=_statement.obj.tokens,
        #                                 attr_tokens=_statement.attr.tokens,
        #                                 value_entity=_statement.attr.name,
        #                                 relation_score=score))
        #         else:
        #             structured_data_list.append(
        #                 types.OAModel(findings_seq=findings_seq,
        #                         obj_entity=_statement.obj,
        #                         attr_entity=_statement.attr,
        #                         relation_score=score))
        #     if self.do_certainty_completion:
        #         structured_data_list += self._certainty_completion(
        #             tokens, obj_entities, findings_seq)
        # else:
        #     if self.do_certainty_completion:
        #         if obj_entities:
        #             structured_data_list += self._certainty_completion(
        #                 tokens, obj_entities, findings_seq)
        return structured_data_list

    def ner(self, report: str) -> types.Tagger:
        if self.do_preprocessing:
            report = self._preprocessing(report)
        if self.do_split_sentence:
            sent_list = self._split_sent(report)
        else:
            sent_list = [report]
        tokens_list, labels_list = list(), list()
        for sent in sent_list:
            if self.do_tokenize:
                tokens = self.mc.parse(sent).strip().split(' ')
            else:
                tokens = sent
            labels = self.tagger.predict(tokens)
            tokens_list.extend(tokens)
            labels_list.extend(labels)
        return types.Tagger(tokens=tokens_list, labels=labels_list)

    def _insert_sep(self, heads_list: List[str], tokens_list: List[List[str]],
                    labels_list: List[List[str]]) -> None:
        """ 文中の区切りにsep_tokenを追加する """
        for i, head in enumerate(heads_list):
            # if head == 'related':
            #     tokens_list[i].insert(0, self.sep_token)  # 文間にSEP記述子を追加
            #     labels_list[i].insert(0, 'O')  # SEPにはOラベルを付与する
            tokens_list[i].append(self.sep_token)
            labels_list[i].append('O')

    def structuring(
            self, report: str) -> Union[List[types.OAModel], List[types.OAVTripletModel]]:
        if self.do_preprocessing:
            report = self._preprocessing(report)
        if self.do_split_sentence:
            sent_list = self._split_sent(report)
        else:
            sent_list = [report]
        tokens_list, labels_list, spc_list = list(), list(), list(
            ['not_related'])
        outputs = list()
        for sent in sent_list:
            if self.do_tokenize:
                tokens = self.mc.parse(sent).strip().split(' ')
            else:
                tokens = sent
            labels = self.tagger.predict(tokens)
            # if self.sc is not None:
            #     is_head = self.sc.predict(tokens)
            #     heads_list.append(is_head)
            tokens_list.append(tokens)
            labels_list.append(labels)
            outputs.append(types.Tagger(tokens=tokens, labels=labels))
        # 文を結合する
        if self.spc is not None:
            if len(outputs) > 1:
                spc_outputs = self.spc.sentence_pair_classification(outputs)
                spc_list.extend(spc_outputs)
                if self.do_insert_sep:
                    self._insert_sep(spc_list, tokens_list, labels_list)
                group_list = self._create_group_idx_list(spc_list)
                tokens_list, labels_list = self._group_sentence(
                    tokens_list, group_list, labels_list)
        structured_data_list = list()
        for i, (tokens, labels) in enumerate(zip(tokens_list, labels_list)):
            _structured_data_list = self._create_structured_data(
                tokens, labels, i)
            structured_data_list.extend(_structured_data_list)
        return types.Structured_Report(structured_data_list=structured_data_list)