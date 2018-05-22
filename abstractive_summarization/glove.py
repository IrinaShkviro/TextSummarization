# -*- coding: utf-8 -*-

import codecs

def load_glove(filename):
    vocab = []
    embd = []
    with codecs.open(filename, 'r', 'utf8') as file:
        for line in file.readlines():
            row = line.strip().split(' ')
            vocab.append(row[0])
            embd.append(row[1:])
        print('GloVe Loaded')
    return vocab, embd