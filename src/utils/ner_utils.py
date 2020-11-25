import os
import json
from typing import List
from pydantic import BaseModel

import torch
import numpy as np
from seqeval.metrics.sequence_labeling import get_entities

from radie.src.model import BiLSTM_CRF

class Mention(BaseModel):
    name: str
    entity: str

class Tagger(object):
    def __init__(self, path):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.word_vocab = torch.load(os.path.join(path, 'ner.vocab'))
        self.label_vocab = torch.load(os.path.join(path, 'ner.label'))
        model_config = json.load(open(os.path.join(path, 'ner.config'), 'r'))
        self.model = BiLSTM_CRF(**model_config)
        # load pre-trained parameters
        self.model.load_state_dict(torch.load(os.path.join(path, 'ner.bin')))
        self.model = self.model.to(self.device)

    def label_itos(self, label_indices):
        return [self.label_vocab.itos[label] for label in label_indices]

    def predict(self, sentence):
        example = [self.word_vocab.stoi[token] for token in sentence]
        example_t = torch.t(torch.LongTensor(example).expand(1,
                                                             -1))  # [words, 1]
        example_t = example_t.to(self.device)
        self.model.eval()
        with torch.no_grad():
            labels = self.model.predict(example_t)[0]
            labels = [self.label_vocab.itos[label] for label in labels]
        return labels


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # return np.exp(x) / np.sum(np.exp(x), axis=axis)sents_itos
    c = np.max(x)
    exp_x = np.exp(x - c)  # for preventing overflow
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x


def _remove_true_labels(label, padnum=1):
    """
    remove padded taggs from real seuences
    """
    if padnum in label:
        endidx = np.where(label == padnum)[0][0]  # head of padnum
        return label[:endidx]
    else:
        return label


def calc_acc(true_labels, pred_labels):
    """
    true_labels : matrix (sentence, batch)
    pred_labels : list [[tag, tag], [tag], ...]
    """
    true_labels = [
        _remove_true_labels(true_label) for true_label in true_labels
    ]
    acc_list = [
        np.sum(true_label == np.array(pred_label)) / true_label.shape[0]
        for true_label, pred_label in zip(true_labels, pred_labels)
    ]
    return np.mean(acc_list)


def sents_itos(sents_labels, vocab, pad=-1, omission=False):
    if omission:
        return [[
            vocab.itos[label][:6] for label in sent_labels if label != pad
        ] for sent_labels in sents_labels]
    else:
        return [[vocab.itos[label] for label in sent_labels if label != pad]
                for sent_labels in sents_labels]


def get_sents(test_iter):
    with torch.no_grad():
        if test_iter.batch_size == 1:
            sent_list = []
            try:
                for test_data in test_iter:
                    sent = test_data.word
                    sent_list.append([idx.item() for idx in sent])
                return sent_list
            except AttributeError as e:
                raise ValueError("error!")

        else:
            for test_data in test_iter:
                sents = test_data.word
            sents = torch.t(sents)
            sents = [list(sent.cpu().numpy()) for sent in sents]
            return sents


def get_true_labels(test_iter, label_name='label', use_flair=False):
    with torch.no_grad():
        if test_iter.batch_size == 1:
            true_labels_list = []
            for i, test_data in enumerate(test_iter):
                if use_flair:
                    true_labels = getattr(test_data[0], label_name)
                else:
                    true_labels = getattr(test_data, label_name)
                true_labels_list.append(
                    [true_label.item() for true_label in true_labels])
            return true_labels_list
        else:
            for i, test_data in enumerate(test_iter):
                if use_flair:
                    true_labels = getattr(test_data[0], label_name)
                else:
                    true_labels = getattr(test_data, label_name)
            true_labels = torch.t(true_labels)
            true_labels = [
                list(_remove_true_labels(true_label).cpu().numpy())
                for true_label in true_labels
            ]
            return true_labels


def sequence_to_csv(filname,
                    docs,
                    pred_labels,
                    true_labels='',
                    encoding='utf-8',
                    header=None,
                    delimiter=' '):
    """
    expected
    docs : [['word','word','word'],['word','word','word']]
    labels : [['label','label','label'],['label','label','label']]
    labels : [['label','label','label'],['label','label','label']]
    """
    import csv
    import codecs

    f = codecs.open(filname, 'w', encoding)
    writer = csv.writer(f, lineterminator='\n', delimiter=delimiter)
    if true_labels:
        if header:
            writer.writerow(['word', 'true', 'predict'])
        for doc, true_label, pred_label in zip(docs, true_labels, pred_labels):
            writer.writerows(list(zip(doc, true_label, pred_label)))
            writer.writerow([])
    else:
        if header:
            writer.writerow(['word', 'predict'])
        for doc, pred_label in zip(docs, pred_labels):
            writer.writerows(list(zip(doc, pred_label)))
            writer.writerow([])
    f.close()


def get_label_name(vocab, omission=False):
    """
    vocab : torchtext.vocab.Vocab
    expected ['O', 'B-Imaging_observation', 'I-Imaging_observation']
    """
    label_name = list(vocab.freqs)
    label_name.remove('O')
    if omission:
        return set([l[2:6] for l in label_name])
    else:
        return set([l[2:] for l in label_name])


def get_target_mentions(tokens: List[str],
                        labels: List[str],
                        target_entity: str = ''):
    mention_list = list()
    entities = get_entities(labels)
    if target_entity:
        entities = list(filter(lambda e: e[0] == target_entity, entities))
    for entity in entities:
        mention = ''.join(tokens[entity[1]:entity[2] + 1])
        mention_list.append((Mention(name=mention, entity=entity[0])))
    return mention_list
