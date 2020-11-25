import json
import random
import os
import logging

import pandas as pd
import numpy as np
import torch
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from radie.src.model import RcModel_Marker, RcModel_Attention
from radie.src.utils import candidate_generation as cg

logger = logging.getLogger("re")


class Rex(object):
    def __init__(self, path):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.word_vocab = torch.load(os.path.join(path, 'rex.vocab'))
        model_config = json.load(open(os.path.join(path, 'rex.config'), 'r'))
        self.model_info = json.load(open(os.path.join(path, 'rex.info'), 'r'))
        if self.model_info['model_name'] == 'lstm':
            self.model = self._load_lstm_model(self.model_info['output_type'],
                                               model_config)
        else:
            raise (ValueError("unexpected model name is specified"))
        # load pre-trained parameters
        self.model.load_state_dict(torch.load(os.path.join(path, 'rex.bin')))
        self.model = self.model.to(self.device)

    def _load_lstm_model(self, output_type, hparams):
        if output_type == 'marker':
            model = RcModel_Marker(**hparams)
        elif output_type == 'attention':
            model = RcModel_Attention(**hparams)
        else:
            raise (ValueError("Unexpected output type is specified"))
        return model

    def predict(self, relation_statements):
        """
        関係ステートメントの関係有無を予測する
        """
        # set input
        toknes_ids = [[
            self.word_vocab.stoi[token] for token in relation_statement.tokens
        ] for relation_statement in relation_statements]
        tokens_tensor = torch.tensor(toknes_ids,
                                     dtype=torch.long).to(self.device)
        masks_ids = [
            relation_statement.entity_masks
            for relation_statement in relation_statements
        ]
        masks_tensor = torch.tensor(masks_ids,
                                    dtype=torch.uint8).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tokens_tensor, masks_tensor)[0]
        probs = F.softmax(logits, dim=1)
        return probs[:, 1].tolist()
        # return pairs, len(pairs) * [0], relation_type_list


# def load_bert_model(cfg, load_trained=False):
#     config = BertConfig.from_json_file(f'{cfg.path.pretrained}config.json')
#     config.output_type = cfg.model.output_type
#     tokenizer = BertTokenizer.from_pretrained(cfg.path.pretrained,
#                                               do_lower_case=False,
#                                               do_basic_tokenize=False)

#     if not load_trained:
#         config.num_labels = 2
#         model = BertForSequenceClassification.from_pretrained(
#             f'{cfg.path.pretrained}pytorch_model.bin', config=config)
#         model.resize_token_embeddings(len(tokenizer))
#     else:
#         model = BertForSequenceClassification(config=config)
#         model.resize_token_embeddings(len(tokenizer))
#         model.load_state_dict(
#             torch.load(f'{cfg.path.model}{cfg.model.name}_model.bin'))

#     return model

# def add_prediction(cfg, pred_list):
#     instance_df = pd.read_json(
#         f'{cfg.path.data}instance_{cfg.model.name}_test.json',
#         orient='records',
#         lines=True)
#     instance_df['pred'] = pred_list
#     instance_df.to_json(
#         f'{cfg.path.result}prediction_{cfg.model.name}_{cfg.model.output_type}.json',
#         force_ascii=False,
#         orient='records',
#         lines=True)
