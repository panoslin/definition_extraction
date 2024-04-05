"""
A class for basic vocab operations.
"""

import os
import pickle
import random

import numpy as np

random.seed(1234)
np.random.seed(1234)


class Vocab:
    def __init__(self, filename, load=False):
        # load from file and ignore all other params
        self.id2word, self.word2id = self.load(filename)
        print("Vocab size {} loaded from file".format(self.size))

    @property
    def size(self):
        return len(self.id2word)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as infile:
            id2word = pickle.load(infile)
            word2id = dict([(id2word[idx], idx) for idx in range(len(id2word))])

        return id2word, word2id

    def save(self, filename):
        if os.path.exists(filename):
            print("Overwriting old vocab file at " + filename)
            os.remove(filename)
        with open(filename, 'wb') as outfile:
            pickle.dump(self.id2word, outfile)
