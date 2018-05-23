# -*- coding: utf-8 -*-

import codecs

import numpy as np

from rouge import rouge_1_doc, rouge_2_doc, rouge_s_doc, Rouge_L

def calc_rouge(data_path, log_path, postfix, method):
    with codecs.open(data_path + 'f_scores' + postfix, 'w', 'utf8') as writer:
        with codecs.open(data_path + log_path, 'r', 'utf8') as reader:
            cur_epoch_num = -1
            cur_epoch_f_score = []
            predicted = False
            actual = False
            gen_summ = ''
            ideal_summ = ''
            lines = reader.readlines()
            n_lines = len(lines)
            cur_line_num = 0
            for line in lines:
                cur_line_num += 1
                print('\r' + str(cur_line_num) + '/' + str(n_lines), end = '')
                if line[0: 11] == 'Iteration: ':
                    if int(line[11:]) == 0:
                        if cur_epoch_num > -1:
                            cur_rouge = np.mean(cur_epoch_f_score, axis = 0)
                            writer.write(str(cur_epoch_num) + ' ' + str(cur_rouge[0]) + ' ' + str(cur_rouge[1]) + ' ' + str(cur_rouge[2]) + '\n')
                        cur_epoch_num += 1
                        cur_epoch_f_score.clear()
                elif line == 'PREDICTED SUMMARY:\n':
                    predicted = True
                elif predicted:
                    predicted = False
                    gen_summ = line
                elif line == 'ACTUAL SUMMARY:\n':
                    actual = True
                elif actual:
                    ideal_summ = line
                    cur_epoch_f_score.append(method(gen_summ, ideal_summ))
                else:
                    continue
                
def calc_rouge_1(data_path, log_path):
    calc_rouge(data_path, log_path, '_rouge_1.txt', rouge_1_doc)
                
def calc_rouge_2(data_path, log_path):
    calc_rouge(data_path, log_path, '_rouge_2.txt', rouge_2_doc)
                
def calc_rouge_s(data_path, log_path):
    calc_rouge(data_path, log_path, '_rouge_s.txt', rouge_s_doc)
                
def calc_rouge_l(data_path, log_path):
    calc_rouge(data_path, log_path, '_rouge_l.txt', Rouge_L.rouge_l)