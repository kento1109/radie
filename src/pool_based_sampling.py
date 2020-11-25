import BiLSTM_CRF

def main():
    BiLSTM_CRF.predict(sentence="結節 よう を",
                       is_printed = False,
                       uncertanity_method = 'least_confident')

if __name__ == '__main__':
    main()