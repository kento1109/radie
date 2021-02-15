from itertools import groupby
from logzero import logger

from radie.src.extractor import Extractor


def main():
    extractor = Extractor(do_preprocessing=True,
                          do_split_sentence=True,
                          do_tokenize=True)

    text = '肺野に１５mm大の結節影を認める。炎症後変化を疑う。肺癌の可能性は低いと思われる。リンパ節腫大は認めない。'

    tagger_result = extractor.ner(text)

    logger.info(f'tagger result : {tagger_result}')

    outputs = extractor(text)
    # outputs = list(map(lambda output: output.chunking(), outputs))
    for output in outputs:
        output.chunking()
    logger.info(f'structured model : {outputs}')


if __name__ == "__main__":
    main()
