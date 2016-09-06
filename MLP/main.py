# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 18:07:39 2016

@author: PeDeNRiQue
"""
import math
import numpy as np


def tangente_hyperbolique(a):
    
    return (math.tanh(a))
    
def multiply_matrix(m, c):
    m = np.array(m)
    c = np.array(c)
    r = m * c
    #print(r)
    return (np.sum(r,axis=1).tolist())
    

if __name__ == "__main__":
    
    number_of_neurons = 3
    number_of_entries = 3
    entries = [-1, 0.3, 0.7]
    
    firts_layer_weights = [[0 for col in range(number_of_entries)] for row in range(number_of_neurons)]
    
    print (multiply_matrix(firts_layer_weights, entries))
    
    
