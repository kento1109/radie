import os
import copy
import json
from typing import List, Optional, Tuple, Union

from collections import defaultdict
from itertools import product
from seqeval.metrics.sequence_labeling import get_entities

from radie.src.lib import types

# global variables

OBJ_NAMES = ['Imaging_observation', 'Clinical_finding']
# ATTR_NAMES = [
#     'Anatomical_entity', 'Certainty_descriptor', 'Size_descriptor',
#     'Characteristics_descriptor', 'Change_descriptor'
# ]
ATTR_NAMES = [
    'Anatomical_entity', 'Size_descriptor', 'Characteristics_descriptor',
    'Change_descriptor'
]
CHANGE_NAME = 'Change_descriptor'

RTYPES = ['obj-attr', 'obj-obj']

ENTITY_MARKERS_INFO = dict()


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


def _insert_entity_tokens(tokens: List[str], pair: Tuple) -> List[str]:
    """
    token配列の所定の位置にmarkerを挿入する
    """
    _tokens = copy.deepcopy(tokens)
    entity_offset = 0
    pair = sorted(pair, key=lambda p: p[1])
    for _entity_info in pair:
        entity_name, start_idx, end_idx = _entity_info
        e_start_token, e_end_token = _build_entity_tokens(entity_name)
        _tokens.insert(start_idx + entity_offset, e_start_token)
        _tokens.insert(end_idx + entity_offset + 2, e_end_token)
        entity_offset += 2
    return _tokens

# def _set_mask_value(token, entity_markers):
#     if token in entity_markers['args1']:
#         return 1
#     elif token in entity_markers['args2']:
#         return 2
#     else:
#         return 0

# def _create_mask(tokens, entity_markers):
#     masks = list(
#         map(lambda token: _set_mask_value(token, entity_markers), tokens))
#     assert (len(list(filter(lambda m: m == 1, masks))) == 2)
#     assert (len(list(filter(lambda m: m == 2, masks))) == 2)
#     return masks

def _get_entity(tokens: List[str], entity_info: Tuple) -> types.Entity:
    entity_name, start_idx, end_idx = entity_info
    target_tokens = tokens[start_idx: end_idx + 1]
    return types.Entity(name=entity_name,
                        tokens=target_tokens,
                        start_idx=start_idx,
                        end_idx=end_idx)


def _get_target_entity(pair, relation_type,
                       tokens) -> Tuple[types.Entity, types.Entity]:
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
    obj_start_idx, obj_end_idx = args1_entity[1], args1_entity[2]
    obj_name, attr_name = args1_entity[0], args2_entity[0]
    obj_tokens = tokens[obj_start_idx:obj_end_idx + 1]
    obj_entity = types.Entity(name=obj_name,
                              tokens=obj_tokens,
                              start_idx=obj_start_idx)
    attr_start_idx, attr_end_idx = args2_entity[1], args2_entity[2]
    attr_tokens = tokens[attr_start_idx:attr_end_idx + 1]
    attr_entity = types.Entity(name=attr_name,
                               tokens=attr_tokens,
                               start_idx=attr_start_idx)
    return obj_entity, attr_entity


def create_relation_statements(
        tagger_result: types.Tagger) -> List[types.RelationStatement]:
    """
    NERの結果から、関係抽出モデルの入力に必要なインスタンスを作成する
    """
    # label sequenceからエンティティ情報を取り出す
    entities = get_entities(tagger_result.labels)
    obj_entities = list(filter(lambda e: e[0] in OBJ_NAMES, entities))
    candidate_statements = list()
    for relation_type in RTYPES:
        pairs = _create_pairs(tagger_result.tokens, entities, relation_type)
        for pair in pairs:
            obj_entity, attr_entity = _get_target_entity(
                pair, relation_type, tagger_result.tokens)
            _tokens = copy.deepcopy(tagger_result.tokens)
            _insert_entity_tokens(_tokens, pair)
            candidate_statements.append(
                types.RelationStatement(tokens=_tokens,
                                        obj=obj_entity,
                                        attr=attr_entity))
    return candidate_statements


def create_object_statement(tokens: List[str], obj_entity: Tuple) -> types.Statement:
    _tokens = copy.deepcopy(tokens)
    start_idx, end_idx = obj_entity[1], obj_entity[2]
    obj_entity = types.Entity(name=obj_entity[0],
                              tokens=_tokens[start_idx:end_idx + 1],
                              start_idx=start_idx,
                              end_idx=end_idx)
    e_start_token, e_end_token = _build_entity_tokens(obj_entity.name)
    _tokens.insert(start_idx, e_start_token)
    _tokens.insert(end_idx + 2, e_end_token)
    return _tokens


def create_entity_statements(
        tagger_result: types.Tagger,
        entity_list: List[Optional[str]] = None) -> List[types.Statement]:
    """
    NERの結果から、Certainty分類モデルの入力に必要なインスタンスを作成する
    """
    statements = list()
    entities = get_entities(tagger_result.labels)
    if entity_list is None:
        entity_list = OBJ_NAMES
    obj_entities = list(filter(lambda e: e[0] in entity_list, entities))
    for obj_e in obj_entities:
        _tokens = copy.deepcopy(tagger_result.tokens)
        _start_idx, _end_idx = obj_e[1], obj_e[2]
        obj_entity = types.Entity(name=obj_e[0],
                                  tokens=_tokens[_start_idx:_end_idx + 1],
                                  start_idx=_start_idx)
        target_entities = [obj_e]
        entity_offset = 0
        target_entities.sort(key=lambda x: (x[1]))  # sort by entity start idx
        for target_e in target_entities:
            entity_name, start_idx, end_idx = target_e
            e_start_token, e_end_token = _build_entity_tokens(entity_name)
            _tokens.insert(start_idx + entity_offset, e_start_token)
            _tokens.insert(end_idx + entity_offset + 2, e_end_token)
            entity_offset += 2
        statements.append(types.Statement(tokens=_tokens, obj=obj_entity))
    return statements


def create_object_statements_from_entities(
        tokens: List[str],
        obj_entities: List[Tuple],
        keyword_list: List[Optional[str]] = None) -> List[str]:
    """
    NERの結果から、Certainty分類モデルの入力に必要なインスタンスを作成する
    """
    _tokens = copy.deepcopy(tokens)
    target_entities = list()
    if keyword_list is not None:
        for obj_e in obj_entities:
            if ''.join(tokens[obj_e[1]:obj_e[2] + 1]) in keyword_list:
                target_entities.append(obj_e)
    else:
        target_entities = obj_entities
    target_entities.sort(key=lambda x: (x[1]))  # sort by entity start idx
    entity_offset = 0
    for target_e in target_entities:
        entity_name, start_idx, end_idx = target_e
        e_start_token, e_end_token = _build_entity_tokens(entity_name)
        _tokens.insert(start_idx + entity_offset, e_start_token)
        _tokens.insert(end_idx + entity_offset + 2, e_end_token)
        entity_offset += 2
    return _tokens