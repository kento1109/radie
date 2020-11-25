import os
import copy
import json
from typing import List

from collections import defaultdict
from pydantic import BaseModel
from itertools import product
from seqeval.metrics.sequence_labeling import get_entities

# global variables

OBJ_NAMES = ['Imaging_observation', 'Clinical_finding']
ATTR_NAMES = [
    'Anatomical_entity', 'Certainty_descriptor', 'Size_descriptor',
    'Characteristics_descriptor', 'Change_descriptor'
]

RTYPES = ['obj-attr', 'obj-obj']

ENTITY_MARKERS_INFO = dict()


class Entity(BaseModel):
    name: str
    tokens: List[str]

    def chunking(self):
        self.tokens = ''.join(self.tokens)

class RelationStatement(BaseModel):
    """関係抽出モデルの入力に必要な情報を保持する"""
    tokens: List[str]
    obj: Entity
    attr: Entity
    entity_masks: List[int]

class ObjectStatement(BaseModel):
    """Certainty分類モデルの入力に必要な情報を保持する"""
    tokens: List[str]
    obj: Entity

def _build_entity_tokens(entity):
    return ['<' + entity + '>', '</' + entity + '>']


def set_marker_info():
    _obj_obj_info = dict()
    _obj_obj_info['args1'] = _build_entity_tokens('Imaging_observation')
    _obj_obj_info['args2'] = _build_entity_tokens('Clinical_finding')
    ENTITY_MARKERS_INFO['obj-obj'] = _obj_obj_info
    _obj_attr_info = defaultdict(list)
    _obj_attr_info['args1'] = _obj_obj_info['args1'] + _obj_obj_info['args2']
    for entity in ATTR_NAMES:
        _obj_attr_info['args2'].extend(_build_entity_tokens(entity))
    ENTITY_MARKERS_INFO['obj-attr'] = _obj_attr_info


def _create_pairs(tokens, entities, relation_type):
    """ 
    指定されたRelation typeのペアを作成する
    （対象のペアがentitiesに存在しない場合、空のリストを返す）
    """
    if relation_type == 'obj-attr':
        # pairs between obj and attr
        args1 = list(filter(lambda e: e[0] in OBJ_NAMES, entities))
        args2 = list(filter(lambda e: e[0] in ATTR_NAMES, entities))
    else:
        # pairs between obj and obj
        args1 = list(filter(lambda e: e[0] == 'Imaging_observation', entities))
        args2 = list(filter(lambda e: e[0] == 'Clinical_finding', entities))
    pairs = list(product(args1, args2))
    return [sorted(pair, key=lambda x: x[1]) for pair in pairs]


def _insert_entity_tokens(tokens, pair):
    """
    token配列の所定の位置にmarkerを挿入する（in-place）
    """
    entity_offset = 0
    for _entity_info in pair:
        entity_name, start_idx, end_idx = _entity_info
        e_start_token, e_end_token = _build_entity_tokens(entity_name)
        tokens.insert(start_idx + entity_offset, e_start_token)
        tokens.insert(end_idx + entity_offset + 2, e_end_token)
        entity_offset += 2


def _set_mask_value(token, entity_markers):
    if token in entity_markers['args1']:
        return 1
    elif token in entity_markers['args2']:
        return 2
    else:
        return 0


def _create_mask(tokens, entity_markers):
    masks = list(
        map(lambda token: _set_mask_value(token, entity_markers), tokens))
    assert (len(list(filter(lambda m: m == 1, masks))) == 2)
    assert (len(list(filter(lambda m: m == 2, masks))) == 2)
    return masks


def _get_target_entity(pair, relation_type, tokens):
    """pairからobj/attrのentityを得る"""
    e1, e2 = pair
    if relation_type == 'obj-attr':
        args1_entity = e1 if e1[0] in OBJ_NAMES else e2
        args2_entity = e1 if e1[0] in ATTR_NAMES else e2
    else:
        args1_entity = e1 if e1[0] == 'Imaging_observation' else e2
        args2_entity = e1 if e1[0] == 'Clinical_finding' else e2
    assert (args1_entity)
    assert (args2_entity)
    obj_name, attr_name = args1_entity[0], args2_entity[0]
    obj_tokens = tokens[args1_entity[1]:args1_entity[2] + 1]
    obj_entity = Entity(name=obj_name, tokens=obj_tokens)
    attr_tokens = tokens[args2_entity[1]:args2_entity[2] + 1]
    attr_entity = Entity(name=attr_name, tokens=attr_tokens)
    return obj_entity, attr_entity


def create_relation_statements(tokens, labels):
    """
    NERの結果から、関係抽出モデルの入力に必要なインスタンスを作成する
    """
    # label sequenceからエンティティ情報を取り出す
    entities = get_entities(labels)
    candidate_statements = list()
    for relation_type in RTYPES:
        pairs = _create_pairs(tokens, entities, relation_type)
        target_entity_markers = ENTITY_MARKERS_INFO[relation_type]
        for pair in pairs:
            obj_entity, attr_entity = _get_target_entity(
                pair, relation_type, tokens)
            _tokens = copy.deepcopy(tokens)
            _insert_entity_tokens(_tokens, pair)
            _masks = _create_mask(_tokens, target_entity_markers)
            candidate_statements.append(
                RelationStatement(tokens=_tokens,
                                  obj=obj_entity,
                                  attr=attr_entity,
                                  entity_masks=_masks))
    return candidate_statements


def create_entity_statements(tokens, labels, entity_list=None, contains_ced_tag=False):
    """
    NERの結果から、Certainty分類モデルの入力に必要なインスタンスを作成する
    """
    object_statements = list()
    entities = get_entities(labels)
    if entity_list is None:
        entity_list = OBJ_NAMES
    obj_entities = list(filter(lambda e: e[0] in entity_list, entities))
    ced_entities = None
    if contains_ced_tag:
         ced_entities = list(filter(lambda e: e[0] == 'Certainty_descriptor', entities))
    for obj_e in obj_entities:
        _tokens = copy.deepcopy(tokens)
        obj_entity = Entity(name=obj_e[0], tokens=_tokens[obj_e[1]: obj_e[2] + 1])
        if contains_ced_tag:
            target_entities = [obj_e] + ced_entities
        else:
            target_entities = [obj_e]
        entity_offset = 0
        target_entities.sort(key=lambda x:(x[1]))  # sort by entity start idx
        for target_e in target_entities:
            entity_name, start_idx, end_idx = target_e
            e_start_token, e_end_token = _build_entity_tokens(entity_name)
            _tokens.insert(start_idx + entity_offset, e_start_token)
            _tokens.insert(end_idx + entity_offset + 2, e_end_token)
            entity_offset += 2
        object_statements.append(ObjectStatement(tokens=_tokens, obj=obj_entity))
    return object_statements


def create_object_statements_from_entities(tokens, obj_entities, keyword_list = None):
    """
    NERの結果から、Certainty分類モデルの入力に必要なインスタンスを作成する
    """
    _tokens = copy.deepcopy(tokens)
    target_entities = list()
    if keyword_list is not None:
        for obj_e in obj_entities:
                if ''.join(tokens[obj_e[1]: obj_e[2] + 1]) in keyword_list:
                    target_entities.append(obj_e)
    else:
        target_entities = obj_entities
    target_entities.sort(key=lambda x:(x[1]))  # sort by entity start idx
    entity_offset = 0
    for target_e in target_entities:
        entity_name, start_idx, end_idx = target_e
        e_start_token, e_end_token = _build_entity_tokens(entity_name)
        _tokens.insert(start_idx + entity_offset, e_start_token)
        _tokens.insert(end_idx + entity_offset + 2, e_end_token)
        entity_offset += 2
    return _tokens