# -*- coding: utf-8 -*-

import collections
import math
import operator
import numpy as np
import os
import shutil
import codecs

from work_with_stories import StoriesCollection

def compute_tfidf(corpus):
    def compute_tf(text):    
        tf_text = collections.Counter(text)    
        for i in tf_text:   
            tf_text[i] = tf_text[i]/float(len(text))
        return tf_text
    
    def compute_idf(word, corpus):    
        return math.log10(len(corpus)/sum([1.0 for i in corpus if word in i]))
    
    documents_list = []
    for text in corpus:
        tf_idf_dictionary = {}
        computed_tf = compute_tf(text)
        for word in computed_tf:
            tf_idf_dictionary[word] = computed_tf[word] * compute_idf(word, corpus)
        documents_list.append(tf_idf_dictionary)
    return documents_list

def ranked_docs(query, corpus, n_top):
    tf_idf_list = compute_tfidf(corpus)
    doc_scores = {}
    for cur_doc in range(len(corpus)):
        cur_doc_score = 0
        for word in query:
            cur_doc_score += tf_idf_list[cur_doc].get(word, 0)
        doc_scores[cur_doc] = cur_doc_score
    doc_scores = sorted(doc_scores.items(), key=operator.itemgetter(1), reverse = True)
    return doc_scores[0 : min(n_top, len(doc_scores))]

if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, "..\cnn")
    db = StoriesCollection(data_dir)
    n_top = 10
    
    for n_top in range(5, 101, 5):
        result_dir = 'results\\extractive_tf_idf_top_%i' % n_top
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir, ignore_errors=True)
        os.mkdir(result_dir)
        os.chdir(result_dir)
          
        n_files = db.get_n_files()
        
        while not db.was_cycle():
            cur_corpus = db.get_next_corpus()
            cur_doc_name = db.get_cur_doc_name()
            header = db.get_header()
            ranked_paragraphs = ranked_docs(header.split(' '), cur_corpus, n_top)
            suitable_paragraphs = db.get_last_paragraphs(np.asarray(ranked_paragraphs, dtype='uint32')[:, 0])
            
            result_string = " ".join(suitable_paragraphs) 
            with codecs.open(cur_doc_name, "w+", "utf_8_sig") as file:
                file.write(result_string)
        os.chdir(cur_dir)
    
