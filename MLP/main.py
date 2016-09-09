# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 18:07:39 2016

@author: PeDeNRiQue
"""
import math
import numpy as np


CONT_LEARNING = 0.5

def tangente_hyperbolique(a):
    
    return (math.tanh(a))

def logistic(value):
    return 1/(1+math.exp(-value))
    
def multiply_matrix(m, c):
    m = np.array(m)
    c = np.array(c)
    r = m * c
    #print(r)
    return (np.sum(r,axis=1).tolist())
    
def activation_function(values):
   values = [ logistic(v) for v in values]
   #print(values)
   return values
    
def error_total(targets, outs):
     result =  [pow((target - out),2)/2 for target,out in zip(targets, outs)]
     return sum(result)   
 
def update_output_layer(targets, outputs,entries, weights):
    
    #print(targets, outputs,entries)
    
    derivative_total_error = [-(target-out) for target,out in zip(targets, outputs)]
    derivative_log_function= [out*(1-out) for out in outputs]
    
    results = [error*log for error,log in zip(derivative_total_error, derivative_log_function)]

    

    for i in range(len(results)):
        for j in range(len(entries)):
            weights[i][j] = weights[i][j] - CONT_LEARNING * (results[i] * entries[j])
            
            
    print(weights)
    
    return weights


if __name__ == "__main__":
    
    
    entries = [1, 0.05, 0.1]
    targets = [0.01, 0.99]
    
    firts_layer_weights = [[0.35, 0.15, 0.20],[0.35, 0.25, 0.30]]
    second_layer_weights= [[0.60, 0.40, 0.45],[0.60, 0.50, 0.55]]
    
    second_layer_entries = [1] + activation_function(multiply_matrix(firts_layer_weights, entries))
    outputs = trirdy_layer_entries = activation_function(multiply_matrix(second_layer_weights, second_layer_entries))   

    #print(error_total(targets, outputs))
    
    second_layer_weights = update_output_layer(targets, outputs,second_layer_entries,second_layer_weights)
    
    #firts_layer_weights = [[0 for col in range(number_of_entries)] for row in range(number_of_neurons)]
        
    #print(second_layer_entries)
    
'''   
if __name__ == "__main__":
    
    
    entries = [-1, 0.3, 0.7]
    
    #firts_layer_weights = [[0 for col in range(number_of_entries)] for row in range(number_of_neurons)]
    firts_layer_weights = [[0.2, 0.4, 0.5],[0.3, 0.6, 0.7], [0.4, 0.8, 0.3]]
    second_layer_weights = [[-0.7, 0.6, 0.2, 0.7],[-0.3, 0.7, 0.2, 0.8]]
    thirdy_layer_weights = [[0.1, 0.8, 0.5]]
    
    second_layer_entries = [-1] + activation_function(multiply_matrix(firts_layer_weights, entries))
    trirdy_layer_entries = [-1] + activation_function(multiply_matrix(second_layer_weights, second_layer_entries))   
    
    output = multiply_matrix(thirdy_layer_weights, trirdy_layer_entries)    
    
    print(output)
    
'''    
    
