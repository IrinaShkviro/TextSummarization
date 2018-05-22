# -*- coding: utf-8 -*-

import pickle
import codecs
import numpy as np

path_vocab = 'prep_data\\vocab_limit'
path_embedding = 'prep_data\\embd_limit'

def add_word_to_vocab(word):
    with open (path_vocab, 'rb') as fp:
        vocab_limit = pickle.load(fp)

    with open (path_embedding, 'rb') as fp:
        embd_limit = pickle.load(fp)
    
    if word not in vocab_limit:
        vocab_limit.append(word)
        embd_limit.append(np.zeros((50),dtype=np.float32))
        
        with codecs.open('prep_data\\vocab_limit', 'wb') as writer:
            pickle.dump(vocab_limit, writer)
        with codecs.open('prep_data\\embd_limit', 'wb') as writer:
            pickle.dump(embd_limit, writer)