from logzero import logger

from radie.src.extractor import Extractor


def main():
    extractor = Extractor(do_preprocessing=True,
                          do_split_sentence=True,
                          do_tokenize=True)

    text = '肺癌術後。肺野に結節影を認める。炎症後変化を疑う。肺癌の可能性は低いと思われる。リンパ節腫大は認めない。'

    tagger_result = extractor.ner(text)

    logger.info(f'tagger result : {tagger_result}')

    relatin_statements = extractor.cg.create_relation_statements(tagger_result)

    logger.info(f'relatin statements : {relatin_statements}')

    # output = extractor(text)

    # logger.info(f'object with certainty scale : {output}')

if __name__ == "__main__":
    main()
