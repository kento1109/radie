import os
import io
import torch
from torchtext.data import Example
from torchtext.utils import unicode_csv_reader
from torchtext import data, datasets
# from flair.embeddings import FlairEmbeddings, StackedEmbeddings
# from flair.data import Sentence

# flair_fw = FlairEmbeddings('/data/sugimoto/test/flair/resources/best-lm_fw.pt')
# flair_bw = FlairEmbeddings('/data/sugimoto/test/flair/resources/best-lm_bw.pt')
#
# flair_embeddings = StackedEmbeddings(embeddings=[FlairEmbeddings('ja-forward'),
#                                                  FlairEmbeddings('ja-backward')])

def get_flair_embedding(minibatch):
    max_length = max([len(example.word) for example in minibatch])

    pre_allocated_zero_tensor = torch.zeros(max_length,
                                            flair_embeddings.embedding_length,
                                            dtype=torch.float,
                                            device="cuda:0")


    sentences = [Sentence(' '.join(example.word)) for example in minibatch]
    flair_embeddings.embed(sentences)
    sentence_embs = []
    sentences_embs = []

    for sentence in sentences:
        seq_length = len(sentence)
        for token in sentence:
            sentence_embs.append(token.get_embedding().unsqueeze(0))
        flair_token_embs = torch.cat(sentence_embs)
        pre_allocated_zero_tensor[:seq_length, :] = flair_token_embs
        sentences_embs.append(pre_allocated_zero_tensor.unsqueeze(0))
        sentence_embs = []

    return torch.transpose(torch.cat(sentences_embs), 0, 1)

class BucketIteratorWithFlair(data.BucketIterator):
    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                yield data.Batch(minibatch, self.dataset, self.device), get_flair_embedding(minibatch)
            if not self.repeat:
                return

class TabularDataset_with_CHAR(data.Dataset):
    def __init__(self, path, format, fields, skip_header=False,
                 csv_reader_params={}, **kwargs):
        format = format.lower()
        make_example = {
            'json': Example.fromJSON, 'dict': Example.fromdict,
            'tsv': Example.fromCSV, 'csv': Example.fromCSV}[format]

        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            if format == 'csv':
                reader = unicode_csv_reader(f, **csv_reader_params)
            elif format == 'tsv':
                reader = unicode_csv_reader(f, delimiter='\t', **csv_reader_params)
            else:
                reader = f

            if format in ['csv', 'tsv'] and isinstance(fields, dict):
                if skip_header:
                    raise ValueError('When using a dict to specify fields with a {} file,'
                                     'skip_header must be False and'
                                     'the file must have a header.'.format(format))
                header = next(reader)
                field_to_index = {f: header.index(f) for f in fields.keys()}
                make_example = partial(make_example, field_to_index=field_to_index)

            if skip_header:
                next(reader)
                
            reader = [[' '.join(line), ' '.join(line)] for line in reader]

            examples = [make_example(line, fields) for line in reader]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(TabularDataset_with_CHAR, self).__init__(examples, fields, **kwargs)

class SequenceTaggingDataset_with_CHAR(data.Dataset):

    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and \
                    not attr.startswith("__"):
                return len(getattr(example, attr))
        return 0

    def __init__(self, path, fields, separator="\t", **kwargs):
        examples = []
        columns = []

        with open(path) as input_file:
            for line in input_file:
                line = line.strip()
                if line == "":
                    if columns:
                        examples.append(data.Example.fromlist(columns, fields))
                    columns = []
                else:
                    seq_list = line.split(separator)
                    seq_list.insert(0, seq_list[0])
                    for i, column in enumerate(seq_list):
                        if len(columns) < i + 1:
                            columns.append([])
                        columns[i].append(column)

            if columns:
                examples.append(data.Example.fromlist(columns, fields))
        super(SequenceTaggingDataset_with_CHAR, self).__init__(examples, fields,
                                                     **kwargs)
        