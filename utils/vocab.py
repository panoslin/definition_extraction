"""
A class for basic vocab operations.
"""
import pickle
from collections import defaultdict

from utils.constant import UNK_ID


class Vocab:
    def __init__(self, filename):
        with open(filename, 'rb') as f:
            self.id2word = pickle.load(f)
            self.word2id = defaultdict(
                lambda: UNK_ID,
                {self.id2word[idx]: idx for idx in range(len(self.id2word))}
            )

    @property
    def size(self):
        return len(self.id2word)
