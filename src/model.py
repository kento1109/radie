import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext import data, datasets
from .torchcrf import CRF
from .torchhelper import FloatTensor, LongTensor, zeros, Attention
from torch.nn import CrossEntropyLoss

# Create model
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim, hidden_dim,
                 batch_size, drop_out_rate=0.5):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim

        # self.hidden_dim = trial.suggest_discrete_uniform('hidden_dim', 128, 256, 128)
        self.drop_out_rate = drop_out_rate

        # if trial is not None:
        #     self.hidden_dim = trial.suggest_discrete_uniform('hidden_dim', 128, 256, 128)
        #     # self.drop_out_rate = trial.suggest_categorical('drop_out_rate', drop_out_rate)
        #     # self.batch_size = trial.suggest_categorical('batch_size', batch_size)
        # else:
        self.hidden_dim = hidden_dim
        self.drop_out_rate = drop_out_rate
        self.batch_size = batch_size

        self.dropout = nn.Dropout(self.drop_out_rate)

        self.vocab_size = vocab_size
        self.num_classes = num_classes

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim,
                                        padding_idx=1)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # self.attention = Attention(hidden_dim)

        # Maps the output of the LSTM into tag space.
        self.emissons = nn.Linear(hidden_dim, self.num_classes)

        self.hidden = self.init_hidden()

        self.crf = CRF(self.num_classes)

    def init_hidden(self):
        return (zeros(2, self.batch_size, self.hidden_dim // 2),
                zeros(2, self.batch_size, self.hidden_dim // 2))

    def _get_lstm_features(self, sentence):
        """
        sentence : (sent, batch)
        """
        # Initialise hidden state
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence)
        if self.drop_out_rate:
            embeds = self.dropout(embeds)

        lstm_out, self.hidden = self.lstm(embeds)

        # lstm_out, _ = self.attention(lstm_out, lstm_out)

        if self.drop_out_rate:
            lstm_out = self.dropout(lstm_out)
        lstm_feats = self.emissons(lstm_out)
        return lstm_feats

    def forward(self, sentence, tags):
        """
        sentence : (sent, batch)
        tags : (sent, batch)
        """
        self.batch_size = sentence.size(1)
        # Get the emission scores from the BiLSTM
        emissions = self._get_lstm_features(sentence)
        # Computing log likelihood
        mask = sentence.ne(1)  # (s, b)
        llh = self.crf(emissions, tags, mask=mask)
        return - llh

    def predict(self, sentence):
        """
        sentence : (sent, batch)
        """
        self.batch_size = sentence.size(1)
        mask = sentence.ne(1)  # (s, b)
        # Get the emission scores from the BiLSTM
        emissions = self._get_lstm_features(sentence)
        return self.crf.decode(emissions, mask=mask)

    def predict_prob(self, sentence, n_best=1):
        """
        sentence : (sent, batch)
        """
        self.batch_size = sentence.size(1)
        mask = sentence.ne(1)  # (s, b)
        # Get the emission scores from the BiLSTM
        emissions = self._get_lstm_features(sentence)
        return self.crf.decode(emissions, mask=mask, prob=True, n_best=n_best)

class BiLSTM_CRF_cls(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim,
                 batch_size, class_num, drop_out_rate=0.5):
        super(BiLSTM_CRF_cls, self).__init__()
        self.embedding_dim = embedding_dim

        # self.hidden_dim = trial.suggest_discrete_uniform('hidden_dim', 128, 256, 128)
        self.drop_out_rate = drop_out_rate

        # if trial is not None:
        #     self.hidden_dim = trial.suggest_discrete_uniform('hidden_dim', 128, 256, 128)
        #     # self.drop_out_rate = trial.suggest_categorical('drop_out_rate', drop_out_rate)
        #     # self.batch_size = trial.suggest_categorical('batch_size', batch_size)
        # else:
        self.hidden_dim = hidden_dim
        self.drop_out_rate = drop_out_rate
        self.batch_size = batch_size

        self.dropout = nn.Dropout(self.drop_out_rate)

        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.class_num = class_num

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim,
                                        padding_idx=1)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        self.attention = Attention(hidden_dim)

        # Maps the output of the LSTM into tag space.
        self.emissons = nn.Linear(hidden_dim, self.tagset_size)

        # linear transformation for sentence classification
        self.sent_cls = nn.Linear(hidden_dim, self.class_num)

        self.hidden = self.init_hidden()

        self.relu = nn.ReLU()

        self.crf = CRF(len(self.tag_to_ix))

    def init_hidden(self):
        return (zeros(2, self.batch_size, self.hidden_dim // 2),
                zeros(2, self.batch_size, self.hidden_dim // 2))

    def _get_lstm_features(self, sentence):
        """
        sentence : (sent, batch)
        """
        # Initialise hidden state
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence)
        if self.drop_out_rate:
            embeds = self.dropout(embeds)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out_t = lstm_out.transpose(0,1)

        attn_out, _ = self.attention(lstm_out_t, lstm_out_t)  # batch * seq_len * hidden_dim

        if self.drop_out_rate:
            lstm_out = self.dropout(lstm_out)
            attn_out = self.dropout(attn_out)

        cls_score = self.sent_cls(attn_out.sum(dim=1))

        cls_score = self.relu(cls_score)

        lstm_feats = self.emissons(lstm_out)
        return lstm_feats, cls_score

    def forward(self, sentence, tags, classes):
        """
        sentence : (sent, batch)
        tags : (sent, batch)
        classes : (sent, batch)
        """
        self.batch_size = sentence.size(1)
        # Get the emission scores from the BiLSTM
        emissions, cls_score = self._get_lstm_features(sentence)
        # Computing log likelihood
        mask = sentence.ne(1)  # (s, b)
        llh = self.crf(emissions, tags, mask=mask)

        cls_loss = F.cross_entropy(cls_score, classes)

        total_loss = - llh + (cls_loss * 100)

        return cls_loss

    def predict(self, sentence):
        """
        sentence : (sent, batch)
        """
        self.batch_size = sentence.size(1)
        mask = sentence.ne(1)  # (s, b)
        # Get the emission scores from the BiLSTM
        emissions, _ = self._get_lstm_features(sentence)
        return self.crf.decode(emissions, mask=mask)

    def predict_cls(self, sentence):
        """
        sentence : (sent, batch)
        """
        self.batch_size = sentence.size(1)
        mask = sentence.ne(1)  # (s, b)
        # Get the emission scores from the BiLSTM
        emissions, cls_score = self._get_lstm_features(sentence)
        return cls_score

    def predict_with_cls(self, sentence):
        """
        sentence : (sent, batch)
        """
        self.batch_size = sentence.size(1)
        mask = sentence.ne(1)  # (s, b)
        # Get the emission scores from the BiLSTM
        emissions, cls_score = self._get_lstm_features(sentence)
        return self.crf.decode(emissions, mask=mask), cls_score

    def predict_prob(self, sentence, n_best=1):
        """
        sentence : (sent, batch)
        """
        self.batch_size = sentence.size(1)
        mask = sentence.ne(1)  # (s, b)
        # Get the emission scores from the BiLSTM
        emissions = self._get_lstm_features(sentence)
        return self.crf.decode(emissions, mask=mask, prob=True, n_best=n_best)

class BiLSTM_CRF_Flair(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim,
                 batch_size, drop_out=False):
        super(BiLSTM_CRF_Flair, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.batch_size = batch_size
        self.drop_out = drop_out

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim,
                                        padding_idx=1)

        self.dropout = nn.Dropout(0.5)

        self.lstm = nn.LSTM(embedding_dim + 4096, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.emissons = nn.Linear(hidden_dim, self.tagset_size)

        self.hidden = self.init_hidden()

        self.crf = CRF(len(self.tag_to_ix))

    def init_hidden(self):
        return (zeros(2, self.batch_size, self.hidden_dim // 2),
                zeros(2, self.batch_size, self.hidden_dim // 2))

    def _get_lstm_features(self, sentence, flair_embeddings):
        """
        sentence : (sent, batch)
        """
        # Initialise hidden state
        self.hidden = self.init_hidden()
        w_embeds = self.word_embeds(sentence)
        embeds = torch.cat((w_embeds, flair_embeddings), 2)
        if self.drop_out:
            embeds = self.dropout(embeds)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        if self.drop_out:
            lstm_out = self.dropout(lstm_out)
        lstm_feats = self.emissons(lstm_out)
        return lstm_feats



    def forward(self, sentence, tags, flair_embeddings):
        """
        sentence : (sent, batch)
        tags : (sent, batch)
        """
        self.batch_size = sentence.size(1)
        # Get the emission scores from the BiLSTM
        emissions = self._get_lstm_features(sentence, flair_embeddings)
        # Computing log likelihood
        mask = sentence.ne(1)  # (s, b)
        llh = self.crf(emissions, tags, mask=mask)
        return - llh

    def predict(self, sentence, flair_embeddings):
        """
        sentence : (sent, batch)
        """
        self.batch_size = sentence.size(1)
        mask = sentence.ne(1)  # (s, b)
        # Get the emission scores from the BiLSTM
        emissions = self._get_lstm_features(sentence, flair_embeddings)
        return self.crf.decode(emissions, mask=mask)

    def predict_prob(self, sentence, n_best=1):
        """
        sentence : (sent, batch)
        """
        self.batch_size = sentence.size(1)
        mask = sentence.ne(1)  # (s, b)
        # Get the emission scores from the BiLSTM
        emissions = self._get_lstm_features(sentence)
        return self.crf.decode(emissions, mask=mask, prob=True, n_best=n_best)
    

class AttnCls(nn.Module):
    def __init__(self, 
                 vocab_size,
                 embedding_dim, 
                 hidden_dim, 
                 num_classes):
        super(AttnCls, self).__init__()
        
        self.hidden_dim = hidden_dim

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(0.5)

        self.attn = Attention(dimensions=hidden_dim)        
        self.lin1 = nn.Linear(embedding_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)        
        self.relu = nn.ReLU()

    def init_hidden(self):
        return (zeros(2, self.batch_size, self.hidden_dim // 2),
                zeros(2, self.batch_size, self.hidden_dim // 2))
        
    def forward(self, word):
        self.batch_size = word.size(0)
        self.hidden = self.init_hidden()

        emb = self.word_embeds(word)
        
        lstm_out, self.hidden = self.lstm(emb, self.hidden)

        lstm_out = self.dropout(lstm_out)
        
        # attn_h, _ = self.attn(lstm_out, lstm_out)
        
        out = self.lin2(lstm_out.sum(dim=1))
        return out

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_num, kernel_sizes, drop_out_rate=0.5):
        super(CNN, self).__init__()

        V = vocab_size
        D = embedding_dim
        C = 2
        Ci = 1  # channels
        Co = kernel_num
        Ks = kernel_sizes
        self.kernel_sizes = kernel_sizes
        self.drop_out_rate = drop_out_rate

        self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

        self.dropout = nn.Dropout(self.drop_out_rate)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, 1, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        if self.drop_out_rate:
            x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit


class BiLSTM_CRF_Char(nn.Module):
    def __init__(self, word_vocab_size, char_vocab_size, tag_to_ix,
                 word_embedding_dim, char_embedding_dim,
                 word_hidden_dim, char_hidden_dim, batch_size, drop_out_rate=0.5):
        super(BiLSTM_CRF_Char, self).__init__()
        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.word_hidden_dim = word_hidden_dim
        self.char_hidden_dim = char_hidden_dim
        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.batch_size = batch_size
        self.drop_out_rate = drop_out_rate

        self.word_embeds = nn.Embedding(self.word_vocab_size, self.word_embedding_dim,
                                        padding_idx=1)

        self.char_embeds = nn.Embedding(self.char_vocab_size, self.char_embedding_dim,
                                        padding_idx=1)

        self.dropout = nn.Dropout(self.drop_out_rate)

        self.char_lstm = nn.LSTM(self.char_embedding_dim, self.char_hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        self.word_lstm = nn.LSTM(self.word_embedding_dim + self.char_hidden_dim, self.word_hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.emissons = nn.Linear(word_hidden_dim, self.tagset_size)

        self.char_hidden = self.init_char_hidden()
        self.word_hidden = self.init_word_hidden()

        self.crf = CRF(len(self.tag_to_ix))

    def init_word_hidden(self):
        return (zeros(2, self.batch_size, self.word_hidden_dim // 2),
                zeros(2, self.batch_size, self.word_hidden_dim // 2))

    def init_char_hidden(self):
        return (zeros(2, self.batch_size, self.char_hidden_dim // 2),
                zeros(2, self.batch_size, self.char_hidden_dim // 2))

    def _get_lstm_features(self, sentence, char):
        """
        sentence : (sent, batch)
        char : (batch, sent, char)
        """
        # Initialise hidden state
        self.word_hidden = self.init_word_hidden()
        self.char_hidden = self.init_char_hidden()
        # char embedding
        char_embeds = self.char_embeds(char)  # [batch, sent, char, embedding_dim]
        char_embeds = char_embeds.transpose(0,1)  # [sent, batch, char, embedding_dim]
        if self.drop_out_rate:
            char_embeds = self.dropout(char_embeds)
        # char LSTM
        char_hidden = []
        for t in range(char_embeds.size(0)):
            curr_char_embeds = char_embeds[t]  # [batch, char, embedding_dim]
            curr_char_embeds = curr_char_embeds.transpose(0,1)  # [char, batch, embedding_dim]
            _, (hn, _) = self.char_lstm(curr_char_embeds, self.char_hidden)  # hn : [2, batch, hidden_size]
            hn_f = hn[0]  # h_n of forward LSTM  [batch, hidden_size]
            hn_b = hn[1]  # h_n of backward LSTM  [batch, hidden_size]
            _hn = torch.cat([hn_f, hn_b], dim=1)  # [batch, hidden_size * 2]
            char_hidden.append(_hn.unsqueeze(0))
        char_hidden = torch.cat(char_hidden, dim=0)  # [sent, batch, hidden_size * 2]

        word_embeds = self.word_embeds(sentence)  # [sent, batch, embedding_dim]

        concat_input = torch.cat((char_hidden, word_embeds), dim=2)

        if self.drop_out_rate:
            concat_input = self.dropout(concat_input)
        lstm_out, self.hidden = self.word_lstm(concat_input, self.word_hidden)
        if self.drop_out_rate:
            lstm_out = self.dropout(lstm_out)
        lstm_feats = self.emissons(lstm_out)
        return lstm_feats

    def forward(self, sentence, char, tags):
        """
        sentence : (sent, batch)
        char : (batch, sent, char)
        tags : (sent, batch)
        """
        self.batch_size = sentence.size(1)
        # Get the emission scores from the BiLSTM
        emissions = self._get_lstm_features(sentence, char)
        # Computing log likelihood
        mask = sentence.ne(1)  # (s, b)
        llh = self.crf(emissions, tags, mask=mask)
        return - llh

    def predict(self, sentence, char):
        """
        sentence : (sent, batch)
        char : (batch, sent, char)
        """
        self.batch_size = sentence.size(1)
        mask = sentence.ne(1)  # (s, b)
        # Get the emission scores from the BiLSTM
        emissions = self._get_lstm_features(sentence, char)
        return self.crf.decode(emissions, mask=mask)

    def predict_prob(self, sentence, char, n_best=1):
        """
        sentence : (sent, batch)
        char : (batch, sent, char)
        """
        self.batch_size = sentence.size(1)
        mask = sentence.ne(1)  # (s, b)
        # Get the emission scores from the BiLSTM
        emissions = self._get_lstm_features(sentence, char)
        return self.crf.decode(emissions, mask=mask, prob=True, n_best=n_best)

def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        This is now written in C in torch.nn.functional
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class RcModel(nn.Module):
    """
    Relation Classification
    """
    def __init__(self, vocab_size, word_embedding_dim, batch_size, 
                 word_hidden_dim, lstm_hidden_dim, drop_out_rate=0.5):
        super(RcModel, self).__init__()

        self.vocab_size = vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.batch_size = batch_size

        self.loss_fct = CrossEntropyLoss()

        self.dropout = nn.Dropout(drop_out_rate)

        self.bilstm = BiLSTM(vocab_size, word_embedding_dim, lstm_hidden_dim, batch_size, drop_out_rate)
        
        self.classifier  = nn.Linear(lstm_hidden_dim, 2)

        self.tanh = nn.Tanh()
        # self.activation = nn.ReLU()
        self.activation = gelu

    def forward(self, input_ids, labels=None):
        """
        input_ids : (batch, seq_length)
        labels : (batch)
        """

        hidden_features = self.bilstm(input_ids)

        last_hidden_features = hidden_features[:, -1, :]  # get last state features

        features = self.activation(last_hidden_features)

        features = self.dropout(features)

        logits = self.classifier(features)  # out : (batch, 2)

        outputs = (logits,)

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs

class RcModel_Marker(nn.Module):
    """
    Relation Classification
    """
    def __init__(self, vocab_size, word_embedding_dim, 
                 word_hidden_dim, batch_size, lstm_hidden_dim, drop_out_rate=0.5):
        super(RcModel_Marker, self).__init__()

        self.vocab_size = vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.word_hidden_dim = word_hidden_dim
        self.batch_size = batch_size

        self.loss_fct = CrossEntropyLoss()

        self.dropout = nn.Dropout(drop_out_rate)

        self.bilstm = BiLSTM(vocab_size, word_embedding_dim, lstm_hidden_dim, batch_size, drop_out_rate)

        self.fc_obj = nn.Linear(lstm_hidden_dim, word_hidden_dim)

        self.fc_attr = nn.Linear(lstm_hidden_dim, word_hidden_dim)

        self.classifier  = nn.Linear(word_hidden_dim * 2, 2)

        self.tanh = nn.Tanh()
        # self.activation = nn.ReLU()
        self.activation = gelu

    def attn_sum(self, x):
        # x : [seq_length, 1, dimensions]
        x = torch.transpose(x, 0, 1)
        out, _ = self.attn(x, x)
        return out.sum(dim=1)

    def forward(self, input_ids, entity_mask, labels=None):
        """
        input_ids : (batch, seq_length)
        entity_mask : (batch, seq_length)
        labels : (batch)
        """

        inputs_feature = self.bilstm(input_ids)

        inputs_feature = self.dropout(inputs_feature)

        batch_size = inputs_feature.size(0)

        obj_list = [out[(mask == 1).nonzero().squeeze()].view(-1, self.lstm_hidden_dim).mean(0).unsqueeze(0) 
                                    for out, mask in zip(inputs_feature, entity_mask)]

        obj = torch.cat(obj_list).view(batch_size, -1)  # [batch, lstm_hidden * 2]

        obj = self.activation(self.fc_obj(obj))

        attr_list = [out[(mask == 2).nonzero().squeeze()].view(-1, self.lstm_hidden_dim).mean(0).unsqueeze(0) 
                                    for out, mask in zip(inputs_feature, entity_mask)]     

        attr = torch.cat(attr_list).view(batch_size, -1)  # [batch, lstm_hidden * 2]

        attr = self.activation(self.fc_attr(attr))

        features = torch.cat([obj, attr], dim=-1)

        logits = self.classifier(features)  # out : (batch, 2)

        outputs = (logits,)

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs

class RcModel_Attention(nn.Module):
    """
    Relation Classification
    """
    def __init__(self, vocab_size, word_embedding_dim, 
                 word_hidden_dim, batch_size, lstm_hidden_dim, drop_out_rate=0.5):
        super(RcModel_Attention, self).__init__()

        self.vocab_size = vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.word_hidden_dim = word_hidden_dim
        self.batch_size = batch_size

        self.loss_fct = CrossEntropyLoss()

        self.dropout = nn.Dropout(drop_out_rate)

        self.bilstm = BiLSTM(vocab_size, word_embedding_dim, lstm_hidden_dim, batch_size, drop_out_rate)

        self.attn = Attention(dimensions=lstm_hidden_dim)

        self.classifier  = nn.Linear(lstm_hidden_dim, 2)

        self.tanh = nn.Tanh()
        # self.activation = nn.ReLU()
        self.activation = gelu

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

        inputs_feature = self.bilstm(input_ids)

        inputs_feature = self.dropout(inputs_feature)

        features = self.attn_sum(inputs_feature)

        features = self.activation(features)

        logits = self.classifier(features)  # out : (batch, 2)

        outputs = (logits,)

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs


class RcBERT(nn.Module):
    """
    Relation Classification using BERT [CLS]
    """
    def __init__(self, vocab_size, word_embedding_dim, 
                 word_hidden_dim, batch_size, lstm_hidden_dim, drop_out_rate=0.5):
        super(RcBERT, self).__init__()

        self.vocab_size = vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.word_hidden_dim = word_hidden_dim
        self.batch_size = batch_size

        self.loss_fct = CrossEntropyLoss()

        self.dropout = nn.Dropout(drop_out_rate)

        self.bilstm = BiLSTM(vocab_size, word_embedding_dim, lstm_hidden_dim, batch_size, drop_out_rate)

        self.attn = Attention(dimensions=lstm_hidden_dim)

        self.classifier  = nn.Linear(lstm_hidden_dim, 2)

        self.tanh = nn.Tanh()
        # self.activation = nn.ReLU()
        self.activation = gelu

    def attn_sum(self, x):
        # x : [batch, seq_length, dimensions]
        # x = torch.transpose(x, 0, 1)
        out, _ = self.attn(x, x)
        return out.sum(dim=1)

    def forward(self, inputs, masks, labels):
        """
        inputs : (batch, seq_length)
        masks : (batch, seq_length)
        labels : (batch)
        """

        inputs_feature = self.bilstm(inputs)

        inputs_feature = self.dropout(inputs_feature)

        features = self.attn_sum(inputs_feature)

        features = self.activation(features)

        out = self.classifier(features)  # out : (batch, 2)

        loss = self.loss_fct(out, labels)

        return out, loss

    def predict(self, inputs, masks):

        inputs_feature = self.bilstm(inputs)

        inputs_feature = self.dropout(inputs_feature)

        features = self.attn_sum(inputs_feature)

        features = self.activation(features)

        out = self.classifier(features)  # out : (batch, 2)

        return out


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 batch_size, drop_out_rate=0.5):
        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim

        self.drop_out_rate = drop_out_rate

        self.hidden_dim = hidden_dim
        self.drop_out_rate = drop_out_rate
        self.batch_size = batch_size

        self.dropout = nn.Dropout(self.drop_out_rate)

        self.vocab_size = vocab_size

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim,
                                        padding_idx=1)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (zeros(2, self.batch_size, self.hidden_dim // 2),
                zeros(2, self.batch_size, self.hidden_dim // 2))

    def _get_lstm_features(self, inputs):
        """
        inputs : (batch, seq_length)
        """
        # Initialise hidden state
        self.hidden = self.init_hidden()
        
        embeds = self.word_embeds(inputs)

        if self.drop_out_rate:
            embeds = self.dropout(embeds)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)

        # lstm_out, _ = self.attention(lstm_out, lstm_out)

        if self.drop_out_rate:
            lstm_out = self.dropout(lstm_out)

        return lstm_out

    def forward(self, inputs):
        """
        inputs : (batch, seq_length)
        """
        self.batch_size = inputs.size(0)
        # Get token representation from the BiLSTM
        outputs = self._get_lstm_features(inputs)

        return outputs

