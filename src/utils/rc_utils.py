from typing import List
from dataclasses import dataclass

from radie.src.utils.classifier_utils import BaseClassifier

@dataclass
class Entity:
    name: str
    entity_idx: int
    start_idx: int
    end_idx: int
    value: str = ''
    misc: str = ''


@dataclass
class Triple:
    obj: Entity
    attr: Entity
    relation: str
    is_ml: bool = False


@dataclass
class EvalRow:
    seq: int
    obj_idx: int
    obj_value: str
    attr_idx: int
    attr_value: str
    relation: str


@dataclass
class InputExample:
    obj: Entity
    attr: Entity
    token_ids: list
    mask_ids: list
    label: int = None


class RelationClassifier(BaseClassifier):
    def __init__(self, path: str, max_batch_size: int):
        super().__init__(path, max_batch_size)

    def predict(self, tokens_list: List[List[str]]) -> List[str]:

        predicted = super().predict(tokens_list, return_as_argmax=True)

        return [self.idx2label[p] for p in predicted]