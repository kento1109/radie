import os
import copy
import json
from typing import List, Optional, Tuple, Union

from collections import defaultdict
from pydantic import BaseModel, Field
from itertools import product
from seqeval.metrics.sequence_labeling import get_entities

from radie.src.utils import types

# global variables

OBJ_NAMES = ['Imaging_observation', 'Clinical_finding']
ATTR_NAMES = [
    'Anatomical_entity', 'Certainty_descriptor', 'Size_descriptor',
    'Characteristics_descriptor', 'Change_descriptor'
]
CHANGE_NAME = 'Change_descriptor'

RTYPES = ['obj-attr', 'obj-obj']

ENTITY_MARKERS_INFO = dict()


class Norm(BaseModel):
    """controlled vocabllary class"""
    # uid: str
    name: str


class Entity(BaseModel):
    name: str = Field(name="entity label name such as Imaging_observation")
    tokens: List[str] = Field(name="mention strings")
    start_idx: int = Field(name="start index of tokens",
                           description="used for specifying entity")
    norms: List[Norm] = None
    

    def chunking(self):
        self.tokens = ''.join(self.tokens)


class Statement(BaseModel):
    """分類モデルの入力に必要な情報を保持する"""
    tokens: List[str]
    obj: Entity


class RelationStatement(Statement):
    """関係抽出モデルの入力に必要な情報を保持する"""
    attr: Entity
    entity_masks: List[int]


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


def _get_target_entity(pair, relation_type, tokens) -> Tuple[Entity, Entity]:
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
    obj_entity = Entity(name=obj_name,
                        tokens=obj_tokens,
                        start_idx=obj_start_idx)
    attr_start_idx, attr_end_idx = args2_entity[1], args2_entity[2]
    attr_tokens = tokens[attr_start_idx:attr_end_idx + 1]
    attr_entity = Entity(name=attr_name,
                         tokens=attr_tokens,
                         start_idx=attr_start_idx)
    return obj_entity, attr_entity


def create_relation_statements(tokens: List[str],
                               labels: List[str]) -> List[RelationStatement]:
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


def create_entity_statements(
        tagger_result: types.Tagger,
        entity_list: List[Optional[str]] = None,
        contains_ced_tag: bool = False) -> List[Statement]:
    """
    NERの結果から、Certainty分類モデルの入力に必要なインスタンスを作成する
    """
    statements = list()
    entities = get_entities(tagger_result.labels)
    if entity_list is None:
        entity_list = OBJ_NAMES
    obj_entities = list(filter(lambda e: e[0] in entity_list, entities))
    ced_entities = None
    if contains_ced_tag:
        ced_entities = list(
            filter(lambda e: e[0] == 'Certainty_descriptor', entities))
    for obj_e in obj_entities:
        _tokens = copy.deepcopy(tagger_result.tokens)
        _start_idx, _end_idx = obj_e[1], obj_e[2]
        obj_entity = Entity(name=obj_e[0],
                            tokens=_tokens[_start_idx:_end_idx + 1],
                            start_idx=_start_idx)
        if contains_ced_tag:
            target_entities = [obj_e] + ced_entities
        else:
            target_entities = [obj_e]
        entity_offset = 0
        target_entities.sort(key=lambda x: (x[1]))  # sort by entity start idx
        for target_e in target_entities:
            entity_name, start_idx, end_idx = target_e
            e_start_token, e_end_token = _build_entity_tokens(entity_name)
            _tokens.insert(start_idx + entity_offset, e_start_token)
            _tokens.insert(end_idx + entity_offset + 2, e_end_token)
            entity_offset += 2
        statements.append(
            Statement(tokens=_tokens, obj=obj_entity))
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