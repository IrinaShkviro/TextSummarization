# -*- coding: utf-8 -*-

import os
import shutil
import codecs

from work_with_stories import StoriesCollection

import pytextrank

def parse_next_doc(corpus, doc_id):
    """
    parse one document to prep for TextRank
    """
    global DEBUG

    base_idx = 0

    for graf_text in pytextrank.filter_quotes(corpus, is_email=False):

        grafs, new_base_idx = pytextrank.parse_graf(0, graf_text, base_idx)
        base_idx = new_base_idx

        for graf in grafs:
            yield graf

stage_1_dir = 'results\\extractive_textrank_stage_1'
stage_2_dir = 'results\\extractive_textrank_stage_2'
stage_3_dir = 'results\\extractive_textrank_stage_3'
stage_4_dir = 'results\\extractive_textrank_stage_4'
  
            
def stage_1():
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, "..\cnn")
    db = StoriesCollection(data_dir)
    
    result_dir = stage_1_dir
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir, ignore_errors=True)
    os.mkdir(result_dir)
    os.chdir(result_dir)
       
    while not db.was_cycle():        
        cur_corpus = db.get_next_corpus_textrank()
        cur_doc_name = db.get_cur_doc_name()
        
        print(cur_doc_name)
        
        with codecs.open(cur_doc_name[:-6] + '.json', "w+", "utf_8_sig") as file:
            for graf in parse_next_doc(cur_corpus, db.get_cur_id()):
                file.write("%s\n" % pytextrank.pretty_print(graf._asdict()))

    os.chdir(cur_dir)

    
def stage_2():   
    cur_dir = os.path.dirname(__file__)
    data_dir = stage_1_dir
    ids = os.listdir(data_dir)
    
    result_dir = stage_2_dir
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir, ignore_errors=True)
    os.mkdir(result_dir)
    os.chdir(result_dir)
    
    if not os.path.exists('pictures'):
        os.mkdir('pictures')

    
    for cur_id in ids:
        if os.path.exists(cur_id):
            continue
        
        cur_file_name = data_dir + "\\" + cur_id
        print(cur_id)
        graph, ranks = pytextrank.text_rank(cur_file_name)
        pytextrank.render_ranks(graph, ranks, cur_id)

        with codecs.open(cur_id, "w+", "utf_8_sig") as file:
            for rl in pytextrank.normalize_key_phrases(cur_file_name, ranks):
                file.write("%s\n" % pytextrank.pretty_print(rl._asdict()))
        
    os.chdir(cur_dir)
    
def stage_3():
    cur_dir = os.path.dirname(__file__)
    data_dir = stage_1_dir
    ids = os.listdir(data_dir)
    
    result_dir = stage_3_dir
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir, ignore_errors=True)
    os.mkdir(result_dir)
    os.chdir(result_dir)
    
    for cur_id in ids:
        print(cur_id)        
        kernel = pytextrank.rank_kernel(stage_2_dir + '\\' + cur_id)
        with codecs.open(cur_id, "w+", "utf_8_sig") as file:
            for s in pytextrank.top_sentences(kernel, stage_1_dir + '\\' + cur_id):
                file.write(pytextrank.pretty_print(s._asdict()))
                file.write("\n")  
    os.chdir(cur_dir)
    
def stage_4():
    cur_dir = os.path.dirname(__file__)
    data_dir = stage_1_dir
    ids = os.listdir(data_dir)
    
    phrase_limit = 15
    word_limit = 100
    result_dir = stage_4_dir + '_limits_' + str(phrase_limit) + '_' + str(word_limit)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir, ignore_errors=True)
    os.mkdir(result_dir)
    os.chdir(result_dir)
    
    for cur_id in ids:
        phrases = ", ".join(set([p for p in pytextrank.limit_keyphrases(stage_2_dir + '\\' + cur_id, phrase_limit=phrase_limit)]))
        sent_iter = sorted(pytextrank.limit_sentences(stage_3_dir + '\\' + cur_id, word_limit=word_limit), key=lambda x: x[1])
        s = []
        
        for sent_text, idx in sent_iter:
            s.append(pytextrank.make_sentence(sent_text))
        
        graf_text = " ".join(s)
        with codecs.open(cur_id[:-5] + '.txt', "w+", "utf_8_sig") as file:
            file.write("**excerpts:** %s\n\n**keywords:** %s" % (graf_text, phrases,))

    os.chdir(cur_dir)

if __name__ == "__main__":
    stage_4()

    
