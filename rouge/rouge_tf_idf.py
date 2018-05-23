# -*- coding: utf-8 -*-

import os
import codecs

from rouge import rouge_1_doc, rouge_2_doc, rouge_s_doc, Rouge_L


def calc_rouge_1_tf_idf(generated_summ_path, ideal_summ_path):
    ids = os.listdir(ideal_summ_path)
    with codecs.open(generated_summ_path + '_rouge_1.txt', 'w', 'utf8') as writer:
        cur_id_num = -1
        n_ids = len(ids)
        for cur_id in ids:
            cur_id_num += 1
            print('\r' + str(cur_id_num) + '/' + str(n_ids), end = '')
            try:
                with codecs.open(generated_summ_path + '\\' + cur_id, 'r', 'utf8') as generate_reader:
                    gen_text = generate_reader.read()
                    with codecs.open(ideal_summ_path + '\\' + cur_id, 'r', 'utf8') as ideal_reader:
                        cur_rouge = rouge_1_doc(gen_text, ideal_reader.read().replace('\n', '. '))
                        writer.write(str(cur_rouge[0]) + ' ' + str(cur_rouge[1]) + ' ' + str(cur_rouge[2]) + '\n')
            except FileNotFoundError :
                continue
            
def calc_rouge_2_tf_idf(generated_summ_path, ideal_summ_path):
    ids = os.listdir(ideal_summ_path)
    with codecs.open(generated_summ_path + '_rouge_2.txt', 'w', 'utf8') as writer:
        cur_id_num = -1
        n_ids = len(ids)
        for cur_id in ids:
            cur_id_num += 1
            print('\r' + str(cur_id_num) + '/' + str(n_ids), end = '')
            try:
                with codecs.open(generated_summ_path + '\\' + cur_id, 'r', 'utf8') as generate_reader:
                    gen_text = generate_reader.read()
                    with codecs.open(ideal_summ_path + '\\' + cur_id, 'r', 'utf8') as ideal_reader:
                        cur_rouge = rouge_2_doc(gen_text, ideal_reader.read().replace('\n', '. '))
                        writer.write(str(cur_rouge[0]) + ' ' + str(cur_rouge[1]) + ' ' + str(cur_rouge[2]) + '\n')
            except FileNotFoundError :
                continue
            
def calc_rouge_s_tf_idf(generated_summ_path, ideal_summ_path):
    ids = os.listdir(ideal_summ_path)
    with codecs.open(generated_summ_path + '_rouge_s.txt', 'w', 'utf8') as writer:
        cur_id_num = -1
        n_ids = len(ids)
        for cur_id in ids:
            cur_id_num += 1
            print('\r' + str(cur_id_num) + '/' + str(n_ids), end = '')
            try:
                with codecs.open(generated_summ_path + '\\' + cur_id, 'r', 'utf8') as generate_reader:
                    gen_text = generate_reader.read()
                    with codecs.open(ideal_summ_path + '\\' + cur_id, 'r', 'utf8') as ideal_reader:
                        cur_rouge = rouge_s_doc(gen_text, ideal_reader.read().replace('\n', '. '))
                        writer.write(str(cur_rouge[0]) + ' ' + str(cur_rouge[1]) + ' ' + str(cur_rouge[2]) + '\n')
            except FileNotFoundError :
                continue
            
def calc_rouge_l_tf_idf(generated_summ_path, ideal_summ_path):
    r = Rouge_L()
    ids = os.listdir(ideal_summ_path)
    with codecs.open(generated_summ_path + '_rouge_l.txt', 'w', 'utf8') as writer:
        cur_id_num = -1
        n_ids = len(ids)
        for cur_id in ids:
            cur_id_num += 1
            print('\r' + str(cur_id_num) + '/' + str(n_ids), end = '')
            try:
                with codecs.open(generated_summ_path + '\\' + cur_id, 'r', 'utf8') as generate_reader:
                    gen_text = generate_reader.read()
                    with codecs.open(ideal_summ_path + '\\' + cur_id, 'r', 'utf8') as ideal_reader:
                        cur_rouge = r.rouge_l([gen_text], [ideal_reader.read().replace('\n', '. ')])
                        writer.write(str(cur_rouge[0]) + ' ' + str(cur_rouge[1]) + ' ' + str(cur_rouge[2]) + '\n')
            except FileNotFoundError :
                continue