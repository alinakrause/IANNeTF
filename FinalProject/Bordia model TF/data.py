"""
Basically just tokenizes text from text files. Maybe use a tf.Tokenizer instead??
And makes one (!) Tensor containing all tokens (in order of occurence).

Sidenote: Differences btw pytorch and tensorflow
Tensor objects (Corpus.train/valid/test) obviously look slightly different:
(pytorch:)    tensor([   0,    1,    1,  ..., 2212, 2213,    1])
(tensorflow:) tf.Tensor([   0    1    1 ... 2212 2213    1], shape=(5649,), dtype=int32)
"""

import os
import tensorflow as tf
from collections import Counter

# makes/contains a word:token dictionary, vocabulary list,
# individual words counter and total words counter
class Dictionary(object):

    def __init__(self):
        self.word2idx = {} # word:token dictionary
        self.idx2word = [] # vocabulary list
        self.counter = Counter() # counts occurences of each word
        self.total = 0 # counts how many words are in text in total


    def add_word(self, word):
        """ Adds new word:token entry to dictionary if not already there,
            updates individual words and total words counters,
            returns assigned token (int)
        """
        if word not in self.word2idx: # if word not already in token dictionary
            self.idx2word.append(word) # append it to vocabulary list
            self.word2idx[word] = len(self.idx2word) - 1 # create word:token entry in dictionary (token = index of vocabulary list)

        token_id = self.word2idx[word]
        self.counter[token_id] += 1 # update occurences of word
        self.total += 1 # update total words count

        return self.word2idx[word]


    def __len__(self):
        """ Returns how large vocabulary currently is.
        """
        return len(self.idx2word)


# makes/containts word:token dictionary for text corpus
# and tensor containing all tokens for training, validation and test set respectively
class Corpus(object):

    # input: path where text files for training, validation and test set are stored
    def __init__(self, path):
        # word:token dictionary (+ vocabulary list and counters)
        self.dictionary = Dictionary()

        # tokenizes text files for training set, validation set and test set
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))


    def tokenize(self, path):
        """ Tokenizes a text file. (The text files should have one sentence per line.)
            Returns Tensor containing all tokens of text file in order of occurence.
        """
        assert os.path.exists(path)
        with open(path, 'r') as f:
            tokens_list = [] # list for storing all tokens for turning into tensor
            for line in f:
                words = line.split() + ['<eos>'] # get list of words with end of sentence marker
                for word in words: # add all words to dictionary
                    token = self.dictionary.add_word(word)
                    tokens_list.append(token) # add tokens to list

        # create tensor containing all tokens of text (in order of occurence)
        ids = tf.constant(tokens_list)

        return ids
