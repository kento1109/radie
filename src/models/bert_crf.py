from typing import List, Optional
from pydantic import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import BertPreTrainedModel, BertModel


class Bert_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert_CRF, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self,
                input_ids,
                attention_mask,
                labels=None):  # dont confuse this with _forward_alg above.
        outputs = self.bert(input_ids, attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emission = self.classifier(sequence_output)
        attention_mask = attention_mask.type(torch.uint8)

        output = self.crf.decode(emission, mask=attention_mask)

        if labels is not None:
            loss = -self.crf(F.log_softmax(emission, 2),
                             labels,
                             mask=attention_mask,
                             reduction='mean')
            return (output, loss)
        else:
            
            return (output)
