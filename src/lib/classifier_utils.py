import os
from typing import List
from typing import Union
from more_itertools import chunked

import numpy as np
import torch
from transformers import BertTokenizer
from transformers import BertConfig

from radie.src.models.custom_bert import BertForSequenceClassification
from radie.src.lib.trainer_utils import to_device


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


class OrdinalConverter():
    def __init__(self, converter_method, num_label):
        self.converter_method = converter_method
        self.num_label = num_label

    def encode(self, y: int):
        if self.converter_method in ('decomposition', 'threshold'):
            y_hat = np.zeros(self.num_label - 1)
            y_hat[:y] = 1
        else:
            y_hat = y
        return y_hat

    def decode(self, y_hat: Union[int, List[float]]):
        if self.converter_method == 'decomposition':
            y_k = np.zeros(self.num_label)
            for k in range(self.num_label):
                if k == 0:
                    y_k[k] = 1 - y_hat[k]
                elif k == (self.num_label - 1):
                    y_k[k] = y_hat[k - 1]
                else:
                    y_k[k] = y_hat[k - 1] - y_hat[k]
            y = np.argmax(y_k)
        elif self.converter_method == 'threshold':
            th = 0.5
            y_k = y_hat > th
            y = np.sum(y_k)
        else:
            y = y_hat
        return y



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


class CertaintyClassifier(BaseClassifier):
    def __init__(self, path: str, max_batch_size: int):
        super().__init__(path, max_batch_size, num_labels=4)

        self.ordinal_converter = OrdinalConverter(
            converter_method='decomposition', num_label=5)

    def predict(self, tokens_list: List[List[str]]) -> List[str]:

        outputs = super().predict(tokens_list)

        outputs_label = list()

        for output in outputs:
            predicted = self.ordinal_converter.decode(output)
            outputs_label.append(self.idx2label[predicted])

        return outputs_label


class RelationClassifier(BaseClassifier):
    def __init__(self, path: str, max_batch_size: int):
        super().__init__(path, max_batch_size)

    def predict(self, tokens_list: List[List[str]]) -> List[str]:

        predicted = super().predict(tokens_list, return_as_argmax=True)

        return [self.idx2label[p] for p in predicted]