"""
Data loader.
"""

import json
import random
from collections import Counter

import numpy as np
import torch

from utils import constant


class DataLoader:
    """
    Load data from json files, preprocess and prepare batches.
    """

    def __init__(self, filename, opt, vocab, evaluation=False):
        self.batch_size = opt['batch_size']
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation

        with open(filename) as f:
            data = self.preprocess(json.load(f), vocab, opt)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]

        self.id2label = dict([(v, k) for k, v in constant.LABEL_TO_ID.items()])
        self.sent_id2label = dict([(v, k) for k, v in constant.SENT_LABEL_TO_ID.items()])
        self.labels = [[self.id2label[l]] for d in data for l in d[-2]]
        self.sent_labels = [self.sent_id2label[d[-1]] for d in data]
        self.size = len(data)

        # chunk into batches
        data = [data[i:i + opt['batch_size']] for i in range(0, len(data), opt['batch_size'])]
        self.data = data
        print(f"{len(data)} batches created for {filename}")

    def preprocess(self, dataset, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        for sentence in dataset:
            tokens = list(sentence['tokens'])
            if opt['lower']:
                tokens = [t.lower() for t in tokens]

            tokens = [vocab.word2id[token] for token in tokens]
            pos = [constant.POS_TO_ID[pos] for pos in sentence['pos']]
            labels = [constant.LABEL_TO_ID[label] for label in sentence['labels']]

            # Parses the dependency head information
            head = [int(x) for x in sentence['heads']]
            # checks for at least one root in the dependency tree
            assert any([x == -1 for x in head])

            # Initializes dependency path representation
            dep_path = [0] * len(sentence['tokens'])
            for i in sentence['dep_path']:
                if i != -1:
                    dep_path[i] = 1

            # Constructs adjacency matrix
            # indicating direct connections between tokens.
            adj = np.zeros((len(sentence['heads']), len(sentence['heads'])))
            for i, h in enumerate(sentence['heads']):
                adj[i][h] = 1
                adj[h][i] = 1

            if self.eval or self.opt['only_label'] != 1 or sentence['label'] != 'none':
                counter = Counter(sentence['labels'])
                terms = [0] * len(sentence['labels'])
                defs = [0] * len(sentence['labels'])
                # Identifies 'Term' and 'Definition' entities within the labels, marking their positions.
                if counter['B-Term'] == 1 and counter['B-Definition'] == 1:
                    for i, label in enumerate(sentence['labels']):
                        if 'Term' in label:
                            terms[i] = 1
                        if 'Definition' in label:
                            defs[i] = 1

                processed.append(
                    (
                        tokens, pos, head, terms, defs, dep_path, adj, labels,
                        constant.SENT_LABEL_TO_ID[sentence['label']])
                )

        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def sent_gold(self):
        return self.sent_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError

        if key < 0 or key >= len(self.data):
            raise IndexError

        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 9

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = self.sort_all(batch, lens)

        # word dropout
        if not self.eval:
            words = [
                self.word_dropout(sent, self.opt['word_dropout'])
                for sent in batch[0]
            ]
        else:
            words = batch[0]

        # convert to tensors
        words = self.get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0)
        pos = self.get_long_tensor(batch[1], batch_size)
        head = self.get_long_tensor(batch[2], batch_size)
        terms = self.get_long_tensor(batch[3], batch_size)
        defs = self.get_long_tensor(batch[4], batch_size)
        dep_path = self.get_long_tensor(batch[5], batch_size).float()
        adj = self.get_float_tensor2D(batch[6], batch_size)

        labels = self.get_long_tensor(batch[7], batch_size)

        sent_labels = torch.FloatTensor(batch[8])

        return words, masks, pos, head, terms, defs, adj, labels, sent_labels, dep_path, orig_idx

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    @staticmethod
    def get_long_tensor(tokens_list, batch_size):
        """ Convert list of list of tokens to a padded LongTensor. """
        token_len = max(len(x) for x in tokens_list)
        tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
        for i, s in enumerate(tokens_list):
            tokens[i, :len(s)] = torch.LongTensor(s)

        return tokens

    @staticmethod
    def get_float_tensor2D(tokens_list, batch_size):
        """ Convert list of list of tokens to a padded LongTensor. """
        token_len = max(len(x) for x in tokens_list)
        tokens = torch.FloatTensor(batch_size, token_len, token_len).fill_(constant.PAD_ID)
        for i, s in enumerate(tokens_list):
            tokens[i, :len(s), :len(s)] = torch.FloatTensor(s)
        return tokens

    @staticmethod
    def sort_all(batch, lens):
        """ Sort all fields by descending order of lens, and return the original indices. """
        unsorted_all = [lens] + [range(len(lens))] + list(batch)
        sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
        return sorted_all[2:], sorted_all[1]

    @staticmethod
    def word_dropout(tokens, dropout):
        """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
        return [
            constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout else x
            for x in tokens
        ]
