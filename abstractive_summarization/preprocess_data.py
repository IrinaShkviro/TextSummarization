# -*- coding: utf-8 -*-

import string
import pickle
import codecs
import numpy as np

from nltk import word_tokenize

from glove import load_glove
from work_with_stories import StoriesCollection


glove_filename = 'glove_data\\glove.6B.50d.txt'

from words_and_vecs import word2vec

class DataPreprocessor(object):
    def __init__(self):
        self.vocab_limit = []
        self.embd_limit = []
        self.vec_summaries = []
        self.vec_texts = []
        self.vocab = []
        self.embedding = []
        self.word_vec_dim = 0
    
    def load_glove(self):
        self.vocab, embd = load_glove(glove_filename)
        
        embedding = np.asarray(embd)
        self.embedding = embedding.astype(np.float32)
        
        self.word_vec_dim = len(embd[0]) # word_vec_dim = dimension of each word vectors
        
    def clean(self, text):
        text = text.lower()
        printable = set(string.printable)
        result = ''.join([word for word in text if word in printable]) # filter strange characters
        return result
    
    def load_stories(self, next_dir=''):
        data_dir = "..\\cnn"
        db = StoriesCollection(data_dir, next_dir)
        ids = db.get_list_of_ids()
        summaries = []
        texts = []
        cur_id_num = 0
        n_ids = len(ids)
        for cur_id in ids:
            cur_id_num += 1
            print('\r' + str(cur_id_num) + '/' + str(n_ids), end = '')
            text, ann = db.get_story(cur_id)
            texts.append(word_tokenize(self.clean(text)))
            summaries.append(word_tokenize(self.clean(ann)))
        return texts, summaries
    
        
    def set_own_vocab_and_embedding(self, texts, summaries):
        # add words from text to vocab_limit and emb_limit
        text_num = 0
        n_texts = len(texts)
        for text in texts:
            text_num += 1
            print('\r' + str(text_num) + '/' + str(n_texts), end = '')
            for word in text:
                if word not in self.vocab_limit:
                    if word in self.vocab:
                        self.vocab_limit.append(word)
                        self.embd_limit.append(word2vec(word, self.embedding, self.vocab))
        print('own vocab for text created\n')
        
        #add words from summaries in vocab_limit and emb_limit                
        sum_num = 0
        n_sums = len(summaries)       
        for summary in summaries:
            sum_num += 1
            print('\r' + str(sum_num) + '/' + str(n_sums), end = '')
            for word in summary:
                if word not in self.vocab_limit:
                    if word in self.vocab:
                        self.vocab_limit.append(word)
                        self.embd_limit.append(word2vec(word, self.embedding, self.vocab))
        print('own vocab for summaries created\n')
                        
        # add special symbols
        if 'unk' not in self.vocab_limit:
            self.vocab_limit.append('unk')
            self.embd_limit.append(word2vec('unk', self.embedding, self.vocab))

        self.vocab_limit.append('<PAD>')
        self.embd_limit.append(np.zeros([self.word_vec_dim]))
        with codecs.open('prep_data\\vocab_limit', 'wb') as writer:
            pickle.dump(self.vocab_limit, writer)
        with codecs.open('prep_data\\embd_limit', 'wb') as writer:
            pickle.dump(self.embd_limit, writer)
     
    def convert_texts_to_vectors(self, texts, dataset):
        text_num = 0
        n_texts = len(texts)
        for text in texts:
            text_num += 1
            print('\r' + str(text_num) + '/' + str(n_texts), end = '')   
            vec_text = []    
            for word in text:
                vec_text.append(word2vec(word, self.embd_limit, self.vocab_limit)) 
            vec_text = np.asarray(vec_text)
            vec_text = vec_text.astype(np.float32)            
            self.vec_texts.append(vec_text)
        print('\ntexts converted to vectors')
        with codecs.open('prep_data\\vec_texts_' + dataset, 'wb') as writer:
            pickle.dump(self.vec_texts, writer)
        
    def convert_summaries_to_vectors(self, summaries, dataset):
        sum_num = 0
        n_sums = len(summaries)       
        for summary in summaries:
            sum_num += 1
            print('\r' + str(sum_num) + '/' + str(n_sums), end = '')  
            vec_summary = []            
            for word in summary:
                vec_summary.append(word2vec(word, self.embd_limit, self.vocab_limit))                                
            vec_summary = np.asarray(vec_summary)
            vec_summary = vec_summary.astype(np.float32)            
            self.vec_summaries.append(vec_summary)
        print('\nsummaries converted to vectors')
        with codecs.open('prep_data\\vec_summaries_' + dataset, 'wb') as writer:
            pickle.dump(self.vec_summaries, writer)
        
    def convert_stories_to_vectors(self, texts, summaries, dataset):
        self.convert_texts_to_vectors(texts, dataset)
        self.convert_summaries_to_vectors(summaries, dataset)

    def preprocess(self, dataset):
        self.load_glove()
        texts, summaries = self.load_stories(dataset)
        self.set_own_vocab_and_embedding(texts, summaries)
        with open ('prep_data\\vocab_limit', 'rb') as fp:
            self.vocab_limit = pickle.load(fp)
        with open ('prep_data\\embd_limit', 'rb') as fp:
            self.embd_limit = pickle.load(fp)
        self.convert_stories_to_vectors(texts, summaries, dataset)