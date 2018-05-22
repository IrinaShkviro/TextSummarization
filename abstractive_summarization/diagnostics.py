# -*- coding: utf-8 -*-

import pickle

path_vec_summs = 'prep_data\\vec_summaries'
path_vec_texts = 'prep_data\\vec_texts'
path_vocab = 'prep_data\\vocab_limit'
path_embedding = 'prep_data\\embd_limit'


with open (path_vocab, 'rb') as fp:
    vocab_limit = pickle.load(fp)

with open (path_embedding, 'rb') as fp:
    embd_limit = pickle.load(fp)
        
with open (path_vec_summs, 'rb') as fp:
    vec_summaries = pickle.load(fp)

with open (path_vec_texts, 'rb') as fp:
    vec_texts = pickle.load(fp)
     
    
count = 0
LEN = 80

for summary in vec_summaries:
    if len(summary) - 1 > LEN:
        count = count + 1
print("Percentage of dataset with summary length beyond "+str(LEN)+": "+str((count/len(vec_summaries))*100)+"% ")

count = 0
D = 10 

window_size = 2*D+1

for text in vec_texts:
    if len(text) < window_size + 1:
        count = count + 1
print("Percentage of dataset with text length less that window size: "+str((count/len(vec_texts))*100)+"% ")

count = 0
LEN = 1300

for text in vec_texts:
    if len(text) > LEN:
        count = count + 1
print("Percentage of dataset with text length more than "+str(LEN)+": "+str((count/len(vec_texts))*100)+"% ")