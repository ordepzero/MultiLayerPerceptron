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