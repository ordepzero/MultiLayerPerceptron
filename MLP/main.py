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

    weights_updated = [[0 for x in range(len(weights[y]))] for y in range(len(weights))] 
    
        
    for i in range(len(results)):
        for j in range(len(entries)):
            weights_updated[i][j] = weights[i][j] - CONT_LEARNING * (results[i] * entries[j])
            

    
    return weights_updated

def update_hidden_layer(targets, outputs, weights,layer_entries,entries,weights_to_update):
    
    derivative_total_error = [-(target-out) for target,out in zip(targets, outputs)]
    derivative_log_function= [out*(1-out) for out in outputs]
    
    results = [error*log for error,log in zip(derivative_total_error, derivative_log_function)]
    
    results_sum = [[weights[x][y]*results[x] for y in range(len(weights[x]))] for x in range(len(weights))]
    results_sum = np.sum(results_sum, axis=0)
    
    
    layer_entries_difference = [entry*(1-entry) for entry in layer_entries]    
    
    results_sum = np.delete(results_sum, 0)
    layer_entries_difference = np.delete(layer_entries_difference, 0)  
    
    temp = [x * y for x,y in zip(results_sum, layer_entries_difference)] 
    temp_mult = [[x*y for y in entries]  for x in temp]
    

    
    weights_updated = [[w-(t*CONT_LEARNING) for w,t in zip(ww,tt)] for ww,tt in zip(weights_to_update,temp_mult)]
    
    return weights_updated

if __name__ == "__main__":
    
    
    entries = [1, 0.05, 0.1]
    targets = [0.01, 0.99]
    
    firts_layer_weights = [[0.35, 0.15, 0.20],[0.35, 0.25, 0.30]]
    second_layer_weights= [[0.60, 0.40, 0.45],[0.60, 0.50, 0.55]]
    
    second_layer_outputs = [1] + activation_function(multiply_matrix(firts_layer_weights, entries))
    outputs = trirdy_layer_outputs = activation_function(multiply_matrix(second_layer_weights, second_layer_outputs))          
    
    second_layer_weights_updated = update_output_layer(targets, outputs,second_layer_outputs,second_layer_weights)

    first_layer_weights_updated = update_hidden_layer(targets, outputs,second_layer_weights,second_layer_outputs,entries,firts_layer_weights)
            
        
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
    
