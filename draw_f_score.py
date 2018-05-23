# -*- coding: utf-8 -*-

import codecs

import numpy as np
import matplotlib.pyplot as plt

def draw_plot(data_path, out_path):
    x, y = get_data(data_path)
    plt.axis([0, np.max(x) - np.min(x), np.min(y), np.max(y)])
    plt.plot(x - np.min(x), y)
    plt.xlabel('epoch') 
    plt.ylabel('f-score')
    plt.title('f-score')
    plt.savefig(out_path, dpi = 1024)
    plt.show()
    
    
def get_data(data_path):
    x = []
    y = []
    with codecs.open(data_path, 'r', 'utf8') as reader:
        for line in reader:
            xy = line.split()
            x.append(int(xy[0])+ 1)
            y.append(float(xy[3]))
    return np.array(x, np.uint32), np.array(y, np.float32)