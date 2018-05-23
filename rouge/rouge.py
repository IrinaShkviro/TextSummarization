# -*- coding: utf-8 -*-

import re
import codecs

import numpy as np

from collections import defaultdict
from itertools import chain

def calc_dict(sents):
    word_count_dict = defaultdict(int)
    for word in re.split('\W+', sents):
        word_count_dict[word] += 1
    return word_count_dict

def calc_bi_dict(sents):
    word_count_dict = defaultdict(int)
    words = re.split('\W+', sents)
    for word_num in range(len(words) - 1):
        word_count_dict[words[word_num] + ' ' + words[word_num + 1]] += 1
    return word_count_dict

def calc_s_dict(sents):
    word_count_dict = defaultdict(int)
    words = re.split('\W+', sents)
    for word_num in range(len(words) - 1):
        for second_word_num in range(word_num + 1, len(words)):
            word_count_dict[words[word_num] + ' ' + words[second_word_num]] += 1
    return word_count_dict

def calc_common_grams(ideal_dict, gen_dict):
    n_matches = 0
    for key, value in ideal_dict.items():
        if key in gen_dict:
            n_matches += min(value, gen_dict[key])
    return n_matches

def calc_ideal_grams(ideal_dict):
    count = 0
    for key, value in ideal_dict.items():
        count += value
    return count

def calc_metrics(ideal_dict, gen_dict):
    common_words = float(calc_common_grams(ideal_dict, gen_dict))
    precision = common_words / calc_ideal_grams(gen_dict)
    recall = common_words / calc_ideal_grams(ideal_dict)
    f_score = 2 * precision * recall / (recall + precision + 1e-7) + 1e-6
    return precision, recall, f_score

def rouge_1_doc(gens, ideals):
    ideal_dict = calc_dict(ideals)
    gen_dict = calc_dict(gens)
    return calc_metrics(ideal_dict, gen_dict)

def rouge_2_doc(gens, ideals):
    ideal_dict = calc_bi_dict(ideals)
    gen_dict = calc_bi_dict(gens)
    return calc_metrics(ideal_dict, gen_dict)
    
def rouge_s_doc(gens, ideals):
    ideal_dict = calc_s_dict(ideals)
    gen_dict = calc_s_dict(gens)
    return calc_metrics(ideal_dict, gen_dict)

def calc_rouge_mean(data_path):   
    prec = []
    rec = []
    f_score = []
    with codecs.open(data_path, 'r', 'utf8') as reader:
        for line in reader.readlines():
            cur_line = line.split()
            prec.append(float(cur_line[0]))
            rec.append(float(cur_line[1]))
            f_score.append(float(cur_line[2]))
    print('\n' + str(np.mean(np.asarray(prec, np.float))) + ' ' + str(np.mean(np.asarray(rec, np.float))) + ' ' + str(np.mean(np.asarray(f_score, np.float))) + '\n')
    
    
def get_unigram_count(tokens):
    count_dict = dict()
    for t in tokens:
        if t in count_dict:
            count_dict[t] += 1
        else:
            count_dict[t] = 1

    return count_dict


class Rouge_L:
    beta = 1

    @staticmethod
    def my_lcs_grid(x, y):
        n = len(x)
        m = len(y)

        table = [[0 for i in range(m + 1)] for j in range(n + 1)]

        for j in range(m + 1):
            for i in range(n + 1):
                if i == 0 or j == 0:
                    cell = (0, 'e')
                elif x[i - 1] == y[j - 1]:
                    cell = (table[i - 1][j - 1][0] + 1, '\\')
                else:
                    over = table[i - 1][j][0]
                    left = table[i][j - 1][0]

                    if left < over:
                        cell = (over, '^')
                    else:
                        cell = (left, '<')

                table[i][j] = cell

        return table

    @staticmethod
    def my_lcs(x, y, mask_x):
        table = Rouge_L.my_lcs_grid(x, y)
        i = len(x)
        j = len(y)

        while i > 0 and j > 0:
            move = table[i][j][1]
            if move == '\\':
                mask_x[i - 1] = 1
                i -= 1
                j -= 1
            elif move == '^':
                i -= 1
            elif move == '<':
                j -= 1

        return mask_x

    @staticmethod
    def rouge_l(cand_sent, ref_sent):
        cand_sents = [cand_sent]
        ref_sents = [ref_sent]
        lcs_scores = 0.0
        cand_unigrams = get_unigram_count(chain(*cand_sents))
        ref_unigrams = get_unigram_count(chain(*ref_sents))
        for cand_sent in cand_sents:
            cand_token_mask = [0 for t in cand_sent]
            cand_len = len(cand_sent)
            for ref_sent in ref_sents:
                Rouge_L.my_lcs(cand_sent, ref_sent, cand_token_mask)

            cur_lcs_score = 0.0
            for i in range(cand_len):
                if cand_token_mask[i]:
                    token = cand_sent[i]
                    if cand_unigrams[token] > 0 and ref_unigrams[token] > 0:
                        cand_unigrams[token] -= 1
                        ref_unigrams[token] -= 1
                        cur_lcs_score += 1

            lcs_scores += cur_lcs_score

        ref_words_count = sum(len(s) for s in ref_sents)
        cand_words_count = sum(len(s) for s in cand_sents)

        precision = lcs_scores / cand_words_count
        recall = lcs_scores / ref_words_count
        f_score = (1 + Rouge_L.beta ** 2) * precision * recall / (recall +
                                                                Rouge_L.beta ** 2 * precision + 1e-7) + 1e-6
        return precision, recall, f_score, lcs_scores