import os
from typing import List

from transformers import BertTokenizer, BertConfig

from radie.src.models.custom_bert import BertForSequenceClassification
from radie.src.utils.trainer_utils import to_tensor


class BaseClassifier(object):
<<<<<<< HEAD
    def __init__(self, path: str, num_labels=0):
=======
    def __init__(self, path: str):
>>>>>>> 45462776677b989047ce27b0aeb685e930dc9e2d

        # load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(path,
                                                       do_basic_tokenize=False)

        # load model
        self.config = BertConfig.from_json_file(
            os.path.join(path, 'config.json'))
        self.idx2label = self.config.id2label
<<<<<<< HEAD
        self.config.num_labels = num_labels if num_labels else len(self.config.id2label)
=======
        self.config.num_labels = len(self.config.id2label)
>>>>>>> 45462776677b989047ce27b0aeb685e930dc9e2d
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

        encoded_dict = to_tensor([encoded_dict])

        output = self.model(**encoded_dict)

        predicted = self.ordinal_converter.decode(
            output.logits.sigmoid().to('cpu').detach().numpy()[0])

        return self.idx2label[predicted]