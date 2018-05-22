# -*- coding: utf-8 -*-

import os
import shutil

source_path = '..\\cnn\\stories'
dest_paths = []
dest_paths.append('..\\cnn\\train\\')
dest_paths.append('..\\cnn\\valid\\')
dest_paths.append('..\\cnn\\test\\')

def divide_data_into_sets():
    ids = os.listdir(source_path)
    cur_id_num = 0
    n_ids = len(ids)
    ends = []
    ends.append(int(n_ids * 0.8))
    ends.append(ends[0] + int(n_ids * 0.15))
    ends.append(n_ids)
    for cur_set in range(3):
        for cur_id_num in range(0 if cur_set == 0 else ends[cur_set - 1], ends[cur_set]):
            print('\r' + str(cur_id_num) + '/' + str(n_ids), end = '')
            shutil.copyfile(source_path + '\\' + ids[cur_id_num], dest_paths[cur_set] + ids[cur_id_num])
            
def divide_train_data():
    ids = os.listdir('..\\cnn\\train\\')
    cur_id_num = 0
    n_ids = len(ids)
    ends = []
    ends.append(int(n_ids * 0.33))
    ends.append(ends[0] + int(n_ids * 0.33))
    ends.append(n_ids)
    for cur_set in range(3):
        for cur_id_num in range(0 if cur_set == 0 else ends[cur_set - 1], ends[cur_set]):
            print('\r' + str(cur_id_num) + '/' + str(n_ids), end = '')
            shutil.copyfile('..\\cnn\\train\\' + ids[cur_id_num], '..\\cnn\\train_' + str(cur_set) + '\\' + ids[cur_id_num])