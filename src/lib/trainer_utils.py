import os
import csv
import copy
import random
from typing import Any, Dict, List, Union

from sklearn.metrics import classification_report
from logzero import logger
import numpy as np
from mojimoji import zen_to_han
import torch


def set_seed(seed: int) -> None:
    """ 学習に使用する乱数を固定する """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def half_width_conversion(
        tokens: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    tokensの全角を半角に変換
    ※カナは全角のまま
    """
    if isinstance(tokens, list):
        tokens = ' '.join(tokens)
        is_list = True
    else:
        is_list = False

    tokens = zen_to_han(tokens, kana=False, ascii=False)
    tokens = tokens.replace("（", "(")
    tokens = tokens.replace("）", ")")
    tokens = tokens.replace("／", "/")
    tokens = tokens.replace("＠", "@")
    tokens = tokens.replace("：", ":")

    if is_list:
        tokens = tokens.split(' ')
    return tokens


def to_tensor(_batch: List):
    """ 入力リストをテンソル型に変換する """
    features = _batch[0].keys()
    batch = dict()
    for f in features:
        batch[f] = torch.tensor([example[f] for example in _batch],
                                dtype=torch.long)
        # batch[f] = torch.tensor([example[f] for example in _batch])
    return batch


def to_device(device, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
    """ 入力バッチを指定デバイスに転送する """
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    return inputs

