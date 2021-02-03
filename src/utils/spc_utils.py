import os
from typing import List, Dict, Any

from more_itertools import windowed
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

from radie.src.utils.trainer_utils import to_device, half_width_conversion
from radie.src.utils import types

class SentencePairClassification():
    def __init__(self, device, path: str):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(path, do_basic_tokenize=False)
        self.config = BertConfig.from_json_file(os.path.join(path, 'config.json'))
        self.model = BertForSequenceClassification.from_pretrained(path, config=self.config)
        self.model.load_state_dict(torch.load(os.path.join(path, 'pytorch_model.bin')))
        self.model.to(self.device)
        self.max_seq_length = 256        

    def _encode_token_pair(self, tokens_a: List[str], tokens_b: List[str]):
        encoded_dict = self.tokenizer(tokens_a, 
                                      tokens_b, 
                                      padding='max_length', 
                                      max_length=self.max_seq_length, 
                                      is_split_into_words=True,
                                      truncation=True,
                                      return_tensors='pt')
        return encoded_dict

    def _make_pair_tokens(self, tokens_list: List[List[str]]) -> List[Any]:
        for tokens_pair in windowed(tokens_list, 2):
            tokens_a = half_width_conversion(tokens_pair[0])
            tokens_b = half_width_conversion(tokens_pair[1])
            yield self._encode_token_pair(tokens_a, tokens_b)

    def sentence_pair_classification(self, ner_outputs: List[types.Tagger]) -> List[str]:
        spc_outputs = list()
        tokens_list = [output.tokens for output in ner_outputs]
        for encoded_token_pair in self._make_pair_tokens(tokens_list):
            output = self.predict(encoded_token_pair)
            spc_outputs.append(output)
        return spc_outputs

    def predict(self, encoded_token_pair) -> str:
        outputs = self.model(**to_device(self.device, encoded_token_pair))
        _, preds = torch.max(outputs.logits, 1)
        return self.config.id2label[preds.item()]