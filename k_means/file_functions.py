# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 19:35:38 2016

@author: PeDeNRiQue
"""

import numpy as np

def put_file_int_array(filename):
    array = []
    
    with open(filename,"r") as f:
        content = f.readlines()
        for line in content: # read rest of lines
            array.append([float(x) for x in line.split()])   
    return np.array(array);
    
    
def read_file(filename):
    array = []
    
    with open(filename,"r") as f:
        content = f.readlines()
        for line in content: # read rest of lines
            array.append([x for x in line.split(",")])   
    return np.array(array);
    
def change_class_name(data,dic):
    for x in range(len(data)):
        data[x][-1] = dic[data[x][-1]]
    return data
    
def str_to_number(data):
    return[[float(j) for j in i] for i in data]
    
def order_data(data):
    #n_class = [1.,2.,3.] #VERSAO ORIGINAL
    n_class = [0.,1.,2.]
    data_alternated = []
    
    P_TRAIN = 0.75
    size_total = len(data)
    size_train = int(size_total*P_TRAIN)
    size_each_class = int(size_train / len(n_class))
    
    #print(size_train,size_each_class)
    
    index = 0
    for x in range(len(data)):   
        c = n_class[index]
        for i in data:            
            if(i[-1] == c):
                #print("IGUAL",i[-1],c,n_class[-1])
                i[-1] = i[-1] * -1
                data_alternated.append(i)
                if(c == n_class[-1]):
                    c = n_class[0]
                    index = -1
                
                index = index + 1
                c = n_class[index]

    data_alternated = np.array(data_alternated)
    data_alternated[:,-1] *= -1
    
    return data_alternated 