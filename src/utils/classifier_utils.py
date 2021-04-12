import os
import sys
from typing import List
from collections import defaultdict
from more_itertools import chunked

import torch
from transformers import BertTokenizer, BertConfig

from radie.src.models.custom_bert import BertForSequenceClassification
from radie.src.utils.trainer_utils import to_tensor, to_device


def get_entity_masks(input_ids, output_type, base_vocab_size, num_entitiies,
                     max_seq_length):
    """
    create masks for specify entity marker positions
    """

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
    def __init__(self, path: str, max_batch_size: int, num_labels: int = 0):

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.max_batch_size = max_batch_size

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

        self.model.to(self.device)

    def predict(self,
                tokens_list: List[List[str]],
                return_as_argmax: bool = False) -> List[str]:

        # restrict batch size to avoid out of memory error
        outputs = list()
        for tokens_batch in chunked(tokens_list, self.max_batch_size):

            encoded_dict = self.tokenizer(
                tokens_batch,
                padding='max_length',
                max_length=self.config.max_position_embeddings,
                truncation=True,
                is_split_into_words=True,
                return_tensors='pt')

            if hasattr(self.config, "output_type"):
                if self.config.output_type.startswith('entity'):
                    entity_mask_list = list()
                    for input_ids in encoded_dict['input_ids']:
                        entity_mask = get_entity_masks(
                            input_ids,
                            output_type=self.config.output_type,
                            base_vocab_size=32000,
                            num_entitiies=self.config.num_entitiies,
                            max_seq_length=self.config.max_position_embeddings)
                        entity_mask_list.append(entity_mask)
                    encoded_dict['entity_mask'] = torch.tensor(
                        entity_mask_list, dtype=torch.long)

            encoded_dict = to_device(self.device, encoded_dict)
            output = self.model(**encoded_dict).logits
            if return_as_argmax:
                outputs.extend(output.argmax(dim=1).tolist())
            else:
                outputs.extend(output.tolist())
            torch.cuda.empty_cache()

        return outputs
