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


def get_entity_masks(input_ids, output_type, base_vocab_size, num_entitiies,
                     max_seq_length):
    # mask for entity markers
    if output_type == 'entity_both':
        entity_mask = list(
            map(lambda idx: 1
                if idx >= base_vocab_size else 0, input_ids))
        n_entity_markers = 4
    elif output_type == 'entity_start':
        entity_mask = list(
            map(
                lambda idx: 1 if (idx >= base_vocab_size) and
                (idx < base_vocab_size + num_entitiies
                 ) else 0, input_ids))
        n_entity_markers = 2
    elif output_type == 'entity_end':
        entity_mask = list(
            map(
                lambda idx: 1 if idx >=
                (base_vocab_size + num_entitiies) else 0,
                input_ids))
        n_entity_markers = 2
    else:
        raise (ValueError('incorrect output type was not specified'))

    if sum(entity_mask) != n_entity_markers:
        # if some entity marker indices are over max sequence length,
        # we use [CLS] representation for classification
        entity_mask = [1] + [0] * (max_seq_length - 1)

    assert len(input_ids) == len(entity_mask)

    return entity_mask


class RelationClassifier(BaseClassifier):
    def __init__(self, path: str):
        super().__init__(path)

    def predict(self, tokens: List[str]):

        output = super().predict(tokens)

        predicted = output.logits.argmax().item()

        return self.idx2label[predicted]