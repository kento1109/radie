import os
from typing import List
from pydantic import BaseModel

import torch
from transformers import BertTokenizer, BertConfig

from radie.src.bert_model import BertForMultiLabelSequenceClassification


class BertExample(BaseModel):
    input_ids: List[int]
    entity_mask: List[int]
    label_ids: List[int] = None


class BertSetting(BaseModel):
    max_seq_length: int
    target_indices: List[int]


def get_entity_masks(input_ids: List[int], target_indices: List) -> List[int]:
    # set 1 for target entity idx
    entity_masks = list(
        map(lambda idx: 1 if idx in target_indices else 0, input_ids))
    return entity_masks


def convert_to_bert_inputs(tokenizer, example, setting) -> BertExample:
    tokens = ' '.join(example['tokens'])
    input_ids = tokenizer(tokens,
                          padding='max_length',
                          max_length=setting.max_seq_length
                          )['input_ids'][:setting.max_seq_length]
    entity_mask = get_entity_masks(input_ids, setting.target_indices)

    return BertExample(input_ids=input_ids,
                       entity_mask=entity_mask,
                       label_ids=example.get('labels'))


def to_tensor(batch):
    input_ids = torch.tensor([example.input_ids for example in batch],
                             dtype=torch.long)
    entity_mask = torch.tensor([example.entity_mask for example in batch],
                               dtype=torch.long)
    label_ids = torch.tensor(
        [example.label_ids for example in batch],
        dtype=torch.float) if batch[0].label_ids is not None else None
    return {
        'input_ids': input_ids,
        'entity_mask': entity_mask,
        'labels': label_ids
    }


class Normalizer():
    def __init__(self, path):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.config = BertConfig.from_json_file(
            os.path.join(path, 'config.json'))
        self.setting = BertSetting(max_seq_length=self.config.max_seq_length,
                                   target_indices=self.config.target_indices)


"""
normalizer for change modifier
"""


class ChangeNormalizer(Normalizer):
    def __init__(self, path):
        super(ChangeNormalizer, self).__init__(path)
        self.model = BertForMultiLabelSequenceClassification.from_pretrained(
            path, config=self.config)

    def normalize(self, tokens: List[str]):
        example = {'tokens': tokens}
        bert_example = convert_to_bert_inputs(self.tokenizer, example,
                                              self.setting)
        inputs = to_tensor([bert_example])
        outputs = self.model(**inputs)[0][0]
        preds = outputs.sigmoid()
        pred_labels = torch.where((preds > 0.5))[0].tolist()
        return [self.config.id2label[i] for i in pred_labels]