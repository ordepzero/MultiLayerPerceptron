# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 19:38:12 2016

@author: PeDeNRiQue
"""

import numpy as np

def normalize_data(f,has_target=True):
    
    x = np.array(f)
    x_normed = (x - x.min(axis=0))/ (x.max(axis=0) - x.min(axis=0))
    
    #SUBSTITUIO OS VALORES ALVO DA ULTIMA COLUNA NOS DADOS NORMALIZADOS
    if(has_target):
        x_normed[:,-1] = f[:,-1]

    return x_normed
    
    
def train_test_data(data,p_train=0.75):
    size_total = len(data)
    size_train = int(size_total*p_train)
    train = data[0:size_train]
    test  = data[size_train:]
    
    return train,test
    
def convert_to_bin_class(value):
    if(value == 1.):
        return [0, 0, 1]
    elif(value == 2.):
        return [0, 1, 0]
    else:
        return [1, 0, 0]