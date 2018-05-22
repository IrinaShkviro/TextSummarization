# -*- coding: utf-8 -*-

import pickle
import codecs

MAX_SUMMARY_LEN = 80
MAX_TEXT_LEN = 1300

D = 10

window_size = 2*D+1

def reduce_data(path_vec_summs, path_vec_texts):
    
    with open ('prep_data\\' + path_vec_summs, 'rb') as fp:
        vec_summs = pickle.load(fp)
        
    with open ('prep_data\\' + path_vec_texts, 'rb') as fp:
        vec_texts = pickle.load(fp)
        
    vec_summaries_reduced = []
    vec_texts_reduced = []
       
    i = 0
    for summary in vec_summs:
        if len(summary) - 1 <= MAX_SUMMARY_LEN \
        and len(vec_texts[i]) >= window_size \
        and len(vec_texts[i]) <= MAX_TEXT_LEN:
            vec_summaries_reduced.append(summary)
            vec_texts_reduced.append(vec_texts[i])
        i=i+1
    
    with codecs.open('prep_data\\reduced_' + path_vec_summs, 'wb') as writer:
        pickle.dump(vec_summaries_reduced, writer)
        
    with codecs.open('prep_data\\reduced_' + path_vec_texts, 'wb') as writer:
        pickle.dump(vec_texts_reduced, writer)