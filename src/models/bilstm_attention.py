import math

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from radie.src.models.bilstm import BiLSTM
from radie.src.torchhelper import Attention


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        This is now written in C in torch.nn.functional
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Bilstm_Attention(nn.Module):
    """
    Relation Classification
    """
    def __init__(self, vocab_size, word_embedding_dim, 
                 batch_size, lstm_hidden_dim, drop_out_rate=0.5):
        super(Bilstm_Attention, self).__init__()

        self.vocab_size = vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.batch_size = batch_size

        self.loss_fct = CrossEntropyLoss()

        self.dropout = nn.Dropout(drop_out_rate)

        self.bilstm = BiLSTM(vocab_size, word_embedding_dim, lstm_hidden_dim, batch_size, drop_out_rate)

        self.attn = Attention(dimensions=lstm_hidden_dim)

        self.classifier  = nn.Linear(lstm_hidden_dim, 2)

        self.tanh = nn.Tanh()
        self.activation = nn.ReLU()
        # self.activation = gelu

    def attn_sum(self, x):
        # x : [batch, seq_length, dimensions]
        # x = torch.transpose(x, 0, 1)
        out, _ = self.attn(x, x)
        return out.sum(dim=1)

    def forward(self, input_ids, labels=None):
        """
        input_ids : (batch, seq_length)
        labels : (batch)
        """

        outputs = dict()

        inputs_feature = self.bilstm(input_ids)

        inputs_feature = self.dropout(inputs_feature)

        features = self.attn_sum(inputs_feature)

        features = self.activation(features)

        logits = self.classifier(features)  # out : (batch, 2)

        outputs['logits'] = logits

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs['loss'] = loss

        return outputs