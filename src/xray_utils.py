import numpy as np
import torch
import execnet
import subprocess
from collections import Counter
from itertools import chain
from scipy.spatial import distance
from sklearn.metrics import cohen_kappa_score
from seqeval.metrics.sequence_labeling import get_entities
import logging

logger = logging.getLogger('tagger')

def logging_init(verbose):
    global logger
    if verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logger.setLevel(log_level)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    logger.addHandler(ch)

    fh = logging.FileHandler('logs/result.log')
    ch.setLevel(log_level)
    logger.addHandler(fh)

    return logger

def data_shuffle(X, y):
    ridx = np.random.randint(999)
    for l in [X, y]:
        random.seed(ridx)
        random.shuffle(l)
        
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        
def remove_true_labels(label, padnum=1):
    """
    remove padded taggs from real seuences
    """
    if padnum in label:
        endidx = np.where(label == padnum)[0][0]  # head of padnum
        return label[:endidx]
    else:
        return label
        


def sents_itos(sents_labels, vocab, pad=-1, omission=False):
    if omission:
        return [[vocab.itos[label][:6] for label in sent_labels if label != pad] for sent_labels in sents_labels]
    else:
        return [[vocab.itos[label] for label in sent_labels if label != pad] for sent_labels in sents_labels]

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

def get_examples(test_iter):
    example_list = []
    for example in test_iter.dataset.examples:
        example_list.append(example.word)
    return example_list

def get_true_labels(test_iter, label_name='label', use_flair=False):
    with torch.no_grad():
        if test_iter.batch_size == 1:
            true_labels_list = []
            for i, test_data in enumerate(test_iter):
                if use_flair:
                    true_labels = getattr(test_data[0], label_name)
                else:
                    true_labels = getattr(test_data, label_name)
                true_labels_list.append([true_label.item() for true_label in true_labels])
            return true_labels_list
        else:
            for i, test_data in enumerate(test_iter):
                if use_flair:
                    true_labels = getattr(test_data[0], label_name)
                else:
                    true_labels = getattr(test_data, label_name)
            true_labels = torch.t(true_labels)
            true_labels = [list(remove_true_labels(true_label).cpu().numpy()) for true_label in true_labels]
            return true_labels

def to_txt(filname, outlist):
    f = open(filname, 'w')
    for out in outlist:
         f.write(out + '\n')
    f.close()

def read_txt(filname, lineterminator='\n'):
    f = open(filname)
    data = f.read()
    f.close()
    return data.split(lineterminator)

def read_conll2txt(filname, delim=' ', n_label=1):
    with open(filname, 'r') as f:
        doc = []
        sent = {}
        sent_word = []
        sent_label = []
        for _ in range(n_label):
            sent_label.append([])
        for i, line in enumerate(f):
            if (line == '\n') or (line == delim + '\n') or (line == delim + delim + '\n'):
                sent['sent'] = sent_word
                for j in range(n_label):
                    sent['label' + str(j)] = sent_label[j]
                doc.append(sent)
                sent = {}
                sent_word = []
                sent_label = []
                for _ in range(n_label):
                    sent_label.append([])
            else:
                line = line.strip()
                row = line.split(delim)
                sent_word.append(row[0])
                for j in range(n_label):
                    sent_label[j].append(row[j+1])
    return doc


def sequence_to_csv(filname, docs, pred_labels, true_labels='', encoding='utf-8', header=None, delimiter=' '):
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
    
def sort_by_freq(inlist, mincnt=0):
    count_dict = Counter(inlist)
    return [key for key,value in count_dict.most_common() if value > mincnt]

def most_similar(word, X, vocab, n=10):
    source_idx =  vocab.stoi[word]
    distances = distance.cdist([X[source_idx]], X, "cosine")[0]
    target_idx = distances.argsort()[1:n+1]
    target_distance = distances[target_idx]
    target_similarity = 1 - target_distance
    return list(zip([vocab.itos[idx] for idx in target_idx], np.round(target_similarity,3)))

def get_target_entities(entities, target):
    return [entity[1:] for entity in entities if entity[0] == target]

def get_chunks(sent, spans):
    return [sent[span[0]:span[1]+1] for span in spans]

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

# def check_wrong_result(docs, examples, true_labels, pred_labels, tag, keyword='', target=True, topn=5, verbose=False):
#     """
#     target == Ture  : True Negative
#               False : False Positive
#     """
#     chunks_list = []
#     words_list = []
#     num_target_chunks = 0
#     num_wrong_chunks = 0
#     for i, (doc, example, true_label, pred_label) in enumerate(zip(docs, examples, true_labels, pred_labels)):
#         true_entities = get_entities(true_label)
#         true_target_spans = get_target_entities(true_entities, tag)
#         num_target_chunks += len(true_target_spans)
#         pred_entities = get_entities(pred_label)
#         pred_target_spans = get_target_entities(pred_entities, tag)
#         if target:
#             tn_spans = list(set(true_target_spans) - set(pred_target_spans))
#             chunks = get_chunks(example, tn_spans)
#             wrong_labels = get_chunks(pred_label, tn_spans)
#             if chunks :
#                 num_wrong_chunks += len(chunks)
#                 for chunk, wrong_label in zip(chunks, wrong_labels):
#                     chunks_list.append('_'.join(chunk))
#                     if keyword != '':
#                         if keyword in chunk:
#                             print(i, [(w,l) for w,l in zip(chunk ,wrong_label)], ' '.join(doc))
#                     else:
#                         if verbose: print(i, [(w,l) for w,l in zip(chunk ,wrong_label)], ' '.join(doc))
#                     for w in chunk:
#                         words_list.append(w)
#         else:
#             fp_spans =  list(set(pred_target_spans) - set(true_target_spans))
#             chunks = get_chunks(example, fp_spans)
#             wrong_labels = get_chunks(true_label, fp_spans)
#             if chunks :
#                 num_wrong_chunks += len(chunks)
#                 for chunk, wrong_label in zip(chunks, wrong_labels):
#                     chunks_list.append('_'.join(chunk))
#                     if keyword != '':
#                         if keyword in chunk:
#                             print(i, [(w,l) for w,l in zip(chunk ,wrong_label)], ' '.join(doc))
#                     else:
#                         if verbose: print(i, [(w,l) for w,l in zip(chunk ,wrong_label)], ' '.join(doc))
#                     for w in chunk:
#                         words_list.append(w)
#     count_words = Counter(words_list)
#     count_chunks = Counter(chunks_list)
#     print('*'*80)
#     print('target : {0}'.format('True Negative' if target else 'False Positive'))
#     print('{} : {:.2f} ({} / {})'.format('Recall' if target else 'Precision', (num_target_chunks-num_wrong_chunks)/num_target_chunks,(num_target_chunks-num_wrong_chunks),num_target_chunks))
#     print('frequent wrong chunks : ')
#     print([chunk for chunk in count_chunks.most_common(topn)])
#     print('frequent wrong words  : ')
#     print([word for word in count_words.most_common(topn)])

def check_wrong_result(doc, true, pred, target_tag):
    has_wrong_result = False
    for token, t_label, p_label in zip(doc, true, pred):
        if ((target_tag in t_label) or (target_tag in p_label)) and (t_label != p_label):
            has_wrong_result = True
    if has_wrong_result:
        print('doc : ' + ' '.join(doc))
        print('true: ' + ' '.join(true))
        print('pred: ' + ' '.join(pred))
        print('-' * 100)


def fleiss_kappa(M):
    N, k = M.shape  # N is # of items, k is # of categories
    n_annotators = float(np.sum(M[0, :]))  # # of annotators

    p = np.sum(M, axis=0) / (N * n_annotators)
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N
    PbarE = np.sum(p * p)

    kappa = (Pbar - PbarE) / (1 - PbarE)

    return kappa

def seq_fleiss_kappa(x, target_tag=''):
    """
    x is expected list of list of list (n_annotaters, n_sentence, n_token)
    """
    xx_flatten_list = []
    tag_set = set(list(chain.from_iterable(list(chain.from_iterable(x)))))
    tag2idx = {tag: i for i, tag in enumerate(tag_set)}
    for xx in x:
        xx_flatten = list(chain.from_iterable(xx))
        if isinstance(xx_flatten[0], str):
            if target_tag:
                xx_flatten = binarize(xx_flatten, target_tag)
            else:
                xx_flatten = [tag2idx[tag] for tag in xx_flatten]
        xx_flatten_list.append(xx_flatten)
    x_arr = np.array(xx_flatten_list)
    x_arrT = x_arr.T
    if target_tag:
        K = 2
    else:
        K = np.max(x_arr) + 1
    M = np.apply_along_axis(lambda x: np.bincount(x, minlength=K), axis=1, arr=x_arrT)
    return fleiss_kappa(M)

def flatten(x):
    return list(chain.from_iterable(x))

def binarize(x, target_tag):
    return [1 if xx.find(target_tag) > -1 else 0 for xx in x]

def seq_cohen_kappa(x1, x2, target_tag=''):
    x1_flatten = flatten(x1)
    x2_flatten = flatten(x2)
    if target_tag:
        x1_flatten = binarize(x1_flatten, target_tag)
        x2_flatten = binarize(x2_flatten, target_tag)
    return cohen_kappa_score(x1_flatten, x2_flatten)

# def call_python_version(Env, Version, Module, Function, ArgumentList):
#     gw = execnet.makegateway("popen//python=%s%s" % (Env, Version))
#     channel= gw.remote_exec("""
#         from %s import %s as the_function
#         channel.send(the_function(*channel.receive()))""" % (Module, Function))
#     channel.send(ArgumentList)
#     channel_receive = channel.receive()
#     channel.close()
#     subprocess.call(["pkill","-KILL","-f","python2.7"])
#     return channel_receive