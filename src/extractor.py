import os
import re
import copy
from typing import List, Union, Tuple
from itertools import chain, groupby, product

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
                 do_normalize=False,
                 do_insert_sep=False,
                 do_certainty_scaling=True,
                 sep_token='。'):

        # load config
        radie_dir = os.path.dirname(__file__)
        self.config = OmegaConf.load(os.path.join(radie_dir, 'config.yaml'))

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.tagger = Tagger(self.config.path.model_ner)
        logger.info(f"tagger loaded ...")

        self.rc = RelationClassifier(
            self.config.path.model_relation_classifier, self.config.max_batch_size)
        logger.info(f"relation classifier model loaded ...")

        # self.change_normalizer = ChangeNormalizer(self.config.path.model_change_normalizer)
        # logger.info(f"change normalizer model loaded ...")

        self.cc = CertaintyClassifier(
            self.config.path.model_certainty_classifier, self.config.max_batch_size)
        logger.info(f"certainty classifier model loaded ...")

        self.cg = candidate_generation
        self.cg.set_marker_info()
        self.do_preprocessing = do_preprocessing
        self.do_split_sentence = do_split_sentence
        self.do_tokenize = do_tokenize
        self.do_certainty_scaling = do_certainty_scaling

        self.do_normalize = do_normalize
        if self.do_tokenize:
            self.mc = MeCab.Tagger(
                f"-Owakati -d {self.config.path.mecab_dict}")
        self.do_insert_sep = do_insert_sep
        self.sep_token = sep_token

    def __call__(self, report: str) -> List[types.StructuredModel]:

        # ner
        tagger_result = self.ner(report)

        # create structured model
        structured_data = self._create_structured_data(tagger_result)

        return structured_data

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

    def _create_structured_data(
            self, tagger_result: types.Tagger) -> List[types.StructuredModel]:
        """
        NERの結果からobject-attributeのペアを作成し、構造化形式で返す
        """
        entities = self.cg.get_entities(tagger_result.labels)
        obj_entities = list(
            filter(lambda entity: entity[0] in self.cg.OBJ_NAMES, entities))
        attr_entities = list(
            filter(lambda entity: entity[0] in self.cg.ATTR_NAMES, entities))
        cf_entities = list(
            filter(lambda entity: entity[0] == 'Clinical_finding', entities))
        structured_models = list()

        object_statement_list = list()
        relation_statement_list = list()
        obj_attr_pairs_list = list()

        for obj_entity in obj_entities:
            object_statement = self.cg.create_object_statement(
                tagger_result.tokens, obj_entity)
            object_statement_list.append(object_statement)

            # pair between obj and attr
            obj_attr_pairs = list(product([obj_entity], attr_entities))
            # obj_attr_pairs = []

            if obj_entity[0] == 'Imaging_observation':
                obj_attr_pairs += list(product([obj_entity], cf_entities))

            obj_attr_pairs_list.append(obj_attr_pairs)

            for obj_attr_pair in obj_attr_pairs:
                relation_statement = self.cg._insert_entity_tokens(
                    tagger_result.tokens, obj_attr_pair)
                relation_statement_list.append(relation_statement)

        # predict each certainty scale
        if self.do_certainty_scaling and object_statement_list:
            certainty_scales = self.cc.predict(object_statement_list)
        # classify relation
        if relation_statement_list:
            relation_labels = self.rc.predict(relation_statement_list)

        num_obj_entities = len(obj_entities)
        relation_idx = 0
        for i in range(num_obj_entities):
            certainty_scale = ''
            if self.do_certainty_scaling:
                certainty_scale = certainty_scales[i]
            clinical_object = types.Object(entity=self.cg._get_entity(
                tagger_result.tokens, obj_entities[i]),
                                           certainty_scale=certainty_scale)
            attr_list = list()
            obj_attr_pairs = obj_attr_pairs_list[i]
            for obj_attr_pair in obj_attr_pairs:
                relation_label = relation_labels[relation_idx]
                if relation_label == 'related':
                    attr_entity = self.cg._get_entity(tagger_result.tokens,
                                                      obj_attr_pair[1])
                    attr_list.append(attr_entity)
                relation_idx += 1
            _structured_model = types.StructuredModel(
                clinical_object=clinical_object, attributes=attr_list)
            structured_models.append(_structured_model)
        return structured_models


    def ner(self, report: str) -> types.Tagger:
        """ named entity recognition module """
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
                tokens = sent.strip().split(' ')
            labels = self.tagger.predict(tokens)
            tokens_list.extend(tokens)
            labels_list.extend(labels)
            if self.do_preprocessing:
                tokens_list.append(self.sep_token)
                labels_list.extend('O')
        torch.cuda.empty_cache()
        return types.Tagger(tokens=tokens_list, labels=labels_list)

