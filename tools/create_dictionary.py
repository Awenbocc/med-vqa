# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         create_dictionary
# Description:  question->word->dictionary for validation & training
# Author:       Boliu.Kelvin
# Date:         2020/4/5
#-------------------------------------------------------------------------------


import os
import pandas as pd
import numpy as np
import _pickle as cPickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        if "? -yes/no" in sentence:
            sentence = sentence.replace("? -yes/no", "")
        if "? -open" in sentence:
            sentence = sentence.replace("? -open", "")
        if "? - open" in sentence:
            sentence = sentence.replace("? - open", "")
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('...', '').replace('x ray', 'x-ray').replace('.', '')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # if a word is not in dictionary, it will be replaced with the last word of dictionary.
                tokens.append(self.word2idx.get(w, self.padding_idx-1))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def create_dictionary(dataroot):
    dictionary = Dictionary()
    questions = []
    files = [
        'VQA-Med-2020-Task1-VQAnswering-TrainVal-Sets/VQAMed2020-VQAnswering-TrainingSet/VQAnswering_2020_Train_QA_pairs.txt', #train dataset
        'VQA-Med-2020-Task1-VQAnswering-TrainVal-Sets/VQAMed2020-VQAnswering-ValidationSet/VQAnswering_2020_Val_QA_Pairs.txt' #validate dateset
    ]
    for path in files:
        qa_pairs = os.path.join(dataroot, path)
        print("processing the {}".format(path))
        raw = pd.read_csv(qa_pairs, sep='|', header=None, names=['id', 'question', 'answer'], index_col=None)
        for id, row in raw.iterrows():
            dictionary.tokenize(row[1], True)     #row[0]: id , row[1]: question , row[2]: answer

    return dictionary

def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':

    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    d = create_dictionary(data)
    d.dump_to_file(data + '/dictionary.pkl')

    d = Dictionary.load_from_file(data + '/dictionary.pkl')
    emb_dim = 300
    glove_file = data + '/glove.6B/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    np.save(data + '/glove6b_init_%dd.npy' % emb_dim, weights)

