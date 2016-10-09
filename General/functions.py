# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 01:01:47 2016

@author: PeDeNRiQue
"""

import math

def func_bipolar(value):
    if(value >= 0):
        return 1
    else:
        return 0
        
def func_sinal(value):
    if(value >= 0):
        return 1
    else:
        return -1
        
def func_tang_hyper(a):
    return (math.tanh(a))

def func_logistic(value):
    return 1/(1+math.exp(-value))   

def func_linear(value):
    return value
    
def hamming_distance(s1, s2):
    assert len(s1) == len(s2)
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))
    
    
print(math.sqrt(1))