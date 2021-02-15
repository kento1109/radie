import os
from typing import List

from transformers import BertTokenizer, BertConfig

from radie.src.models.custom_bert import BertForSequenceClassification
from radie.src.utils.trainer_utils import to_tensor


def get_entity_masks(input_ids, output_type, base_vocab_size, num_entitiies,
                     max_seq_length):

    # mask for entity markers
    if output_type == 'entity_both':
        entity_mask = list(
            map(lambda idx: 1 if idx >= base_vocab_size else 0, input_ids))
        n_entity_markers = 4
    elif output_type == 'entity_start':
        entity_mask = list(
            map(
                lambda idx: 1 if (idx >= base_vocab_size) and
                (idx < base_vocab_size + num_entitiies) else 0, input_ids))
        n_entity_markers = 2
    elif output_type == 'entity_end':
        entity_mask = list(
            map(
                lambda idx: 1
                if idx >= (base_vocab_size + num_entitiies) else 0, input_ids))
        n_entity_markers = 2
    else:
        raise (ValueError('incorrect output type was not specified'))

    if sum(entity_mask) != n_entity_markers:
        # if some entity marker indices are over max sequence length,
        # we use [CLS] representation for classification
        entity_mask = [1] + [0] * (max_seq_length - 1)

    assert len(input_ids) == len(entity_mask)

    return entity_mask


class BaseClassifier(object):
    def __init__(self, path: str, num_labels=0):

        # load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(path,
                                                       do_basic_tokenize=False)

        # load model
        self.config = BertConfig.from_json_file(
            os.path.join(path, 'config.json'))
        self.idx2label = self.config.id2label
        self.config.num_labels = num_labels if num_labels else len(
            self.config.id2label)
        # load model
        self.model = BertForSequenceClassification.from_pretrained(
            path, config=self.config)

    def predict(self, tokens: List[str]) -> str:

        encoded_dict = self.tokenizer(
            tokens,
            padding='max_length',
            max_length=self.config.max_position_embeddings,
            truncation=True,
            is_split_into_words=True)

        if hasattr(self.config, "output_type"):
            if self.config.output_type.startswith('entity'):
                encoded_dict['entity_mask'] = get_entity_masks(
                    encoded_dict['input_ids'],
                    output_type=self.config.output_type,
                    base_vocab_size=32000,
                    num_entitiies=self.config.num_entitiies,
                    max_seq_length=self.config.max_position_embeddings)

        encoded_dict = to_tensor([encoded_dict])

        output = self.model(**encoded_dict)

        return output