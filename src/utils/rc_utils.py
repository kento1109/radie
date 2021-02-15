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
    def __init__(self, path: str):
        super().__init__(path)

    def predict(self, tokens: List[str]) -> str:

        output = super().predict(tokens)

        predicted = output.logits.argmax().item()

        return self.idx2label[predicted]