"""
A class for basic vocab operations.
"""
import os
import pickle
from collections import defaultdict

from utils.constant import UNK_ID


class Vocab:
    def __init__(self, filename, load=False):
        # load from file and ignore all other params
        self.id2word, self.word2id = self.load(filename)

    @property
    def size(self):
        return len(self.id2word)

    @staticmethod
    def load(filename):
        """
        loads a vocabulary from a file
        """
        with open(filename, 'rb') as infile:
            id2word = pickle.load(infile)
            word2id = defaultdict(
                lambda: UNK_ID,
                {id2word[idx]: idx for idx in range(len(id2word))}
            )

        return id2word, word2id

    def save(self, filename):
        if os.path.exists(filename):
            print("Overwriting old vocab file at " + filename)
            os.remove(filename)
        with open(filename, 'wb') as outfile:
            pickle.dump(self.id2word, outfile)
