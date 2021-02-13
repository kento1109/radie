from typing import List, Any, Dict, Union

import numpy as np

from radie.src.utils.classifier_utils import BaseClassifier


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


class CertaintyClassifier(BaseClassifier):
    def __init__(self, path: str):
        super().__init__(path)

        self.ordinal_converter = OrdinalConverter(
            converter_method='decomposition', num_label=5)

    def predict(self, tokens: List[str]):

        output = super().predict(tokens)

        predicted = self.ordinal_converter.decode(
            output.logits.sigmoid().to('cpu').detach().numpy()[0])

        return self.idx2label[predicted]
