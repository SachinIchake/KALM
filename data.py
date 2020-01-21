import os
import torch

from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()

        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        maxSentenceSize = 0
        SentenceSize =0
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                if line == '\n':
                    self.dictionary.add_word('<eos>')
                    # tokens += 1
                    # print(maxSentenceSize )
                    if maxSentenceSize > SentenceSize:
                        SentenceSize = maxSentenceSize
                    maxSentenceSize = 0
                else:
                    words = line.split()
                    tokens += 1
                    self.dictionary.add_word(words[0])
                    maxSentenceSize += 1


        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                if line == '\n':
                    ids[token] = self.dictionary.word2idx['<eos>']
                    tokens += 1
                else:
                    words = line.split()
                    ids[token] = self.dictionary.word2idx[words[0]]
                    token += 1

        print(SentenceSize)
        return ids

#
# import os
# import torch
#
# from collections import Counter
#
#
# class Dictionary(object):
#     def __init__(self):
#         self.word2idx = {}
#         self.idx2word = []
#         self.counter = Counter()
#         self.total = 0
#
#     def add_word(self, word):
#         if word not in self.word2idx:
#             self.idx2word.append(word)
#             self.word2idx[word] = len(self.idx2word) - 1
#         token_id = self.word2idx[word]
#         self.counter[token_id] += 1
#         self.total += 1
#         return self.word2idx[word]
#
#     def __len__(self):
#         return len(self.idx2word)
#
#
# class Corpus(object):
#     def __init__(self, path):
#         self.dictionary = Dictionary()
#         self.train = self.tokenize(os.path.join(path, 'train.txt'))
#         self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
#         self.test = self.tokenize(os.path.join(path, 'test.txt'))
#
#     def tokenize(self, path):
#         """Tokenizes a text file."""
#         assert os.path.exists(path)
#         # Add words to the dictionary
#         with open(path, 'r') as f:
#             tokens = 0
#             for line in f:
#                 words = line.split() + ['<eos>']
#                 tokens += len(words)
#                 for word in words:
#                     self.dictionary.add_word(word)
#
#         # Tokenize file content
#         with open(path, 'r') as f:
#             ids = torch.LongTensor(tokens)
#             token = 0
#             for line in f:
#                 words = line.split() + ['<eos>']
#                 for word in words:
#                     ids[token] = self.dictionary.word2idx[word]
#                     token += 1
#
#         return ids
