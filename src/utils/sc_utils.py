import os
import json
import logging

import torch
import torch.nn.functional as F

from radie.src.model import CNN

logger = logging.getLogger("sc")


class SC(object):
    def __init__(self, path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word_vocab = torch.load(os.path.join(path, 'sc.vocab'))
        model_config = json.load(open(os.path.join(path, 'sc.config'), 'r'))
        self.model = CNN(**model_config)
        # load pre-trained parameters
        self.model.load_state_dict(torch.load(os.path.join(path, 'sc.bin')))
        self.model = self.model.to(self.device)

    def predict(self, sentence):
        example = [self.word_vocab.stoi[token] for token in sentence]
        padding(self.model, self.word_vocab, example)
        example = torch.LongTensor(example).expand(1, -1)  # [words, 1]
        example = example.to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model.forward(example)[0]
        pred = torch.argmax(logits)
        # probs = F.softmax(logits, dim=0)
        return pred.item()  # get the probability of the head sentence

def padding(model, vocab, sents):
    """
    sent size must be longer than max kernel size -1
    """
    min_length = model.kernel_sizes[-1]
    padidx = vocab.stoi['<pad>']
    while min_length > len(sents):
        sents.append(padidx)