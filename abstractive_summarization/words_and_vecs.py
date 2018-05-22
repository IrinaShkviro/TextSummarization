# -*- coding: utf-8 -*-

import numpy as np
   
def np_nearest_neighbour(x, embedding):
    xdoty = np.multiply(embedding, x)
    xdoty = np.sum(xdoty, 1)
    xlen = np.square(x)
    xlen = np.sum(xlen, 0)
    xlen = np.sqrt(xlen)
    ylen = np.square(embedding)
    ylen = np.sum(ylen, 1)
    ylen = np.sqrt(ylen)
    xlenylen = np.multiply(xlen, ylen)
    cosine_similarities = np.divide(xdoty, xlenylen)
    return embedding[np.argmax(cosine_similarities)]

def word2vec(word, embedding, vocab):
    if word in vocab:
        return embedding[vocab.index(word)]
    else:
        return embedding[vocab.index('unk')]

def vec2word(vec, embedding, vocab):
    for x in range(0, len(embedding)):
        if np.array_equal(embedding[x], np.asarray(vec)):
            return vocab[x]
    return vec2word(np_nearest_neighbour(np.asarray(vec), embedding), embedding, vocab)