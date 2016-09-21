# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 11:37:05 2016

@author: PeDeNRiQue
"""

import math
import numpy as np


CONT_LEARNING = 0.5 
N_NEURONE_FIRST_LAYER = 4
N_NEURONE_SECON_LAYER = 5

weights_input = []#MATRIZ DE PESOS DA CAMADA DE ENTRADA
weights_inter = []#MATRIZ DE PESOS DA CAMADA INTERMEDIATIA
weights_outpt = []#MATRIZ DE PESOS DA CAMADA DE SAIDA

def normalize_data(f,has_target=True):
    
    x = np.array(f)
    x_normed = (x - x.min(axis=0))/ (x.max(axis=0) - x.min(axis=0))
    
    #SUBSTITUIO OS VALORES ALVO DA ULTIMA COLUNA NOS DADOS NORMALIZADOS
    if(has_target):
        x_normed[:,-1] = f[:,-1]

    return x_normed

def put_file_int_array(filename):
    array = []
    
    with open(filename,"r") as f:
        content = f.readlines()
        for line in content: # read rest of lines
            array.append([float(x) for x in line.split()])   
    return np.array(array);

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
    
    #DELTA = -(TARGET - OUT) * OUT*(1 - OUT)
    derivative_total_error = [-(target-out) for target,out in zip(targets, outputs)]
    derivative_log_function= [out*(1-out) for out in outputs]
    delta = [error*log for error,log in zip(derivative_total_error, derivative_log_function)]
    #print("-> ",targets," ", outputs," ",entries," ", weights)
    #print("DELTA:",delta)
    
    weights_updated = [[0 for x in range(len(weights[y]))] for y in range(len(weights))] 
    
        
    for i in range(len(delta)):
        for j in range(len(entries)):
            #print("< ", weights[i][j] , CONT_LEARNING , delta[i] , entries[j]," >")
            weights_updated[i][j] = weights[i][j] + CONT_LEARNING * (-delta[i] * entries[j])
            
            
    return weights_updated,delta

def update_hidden_layer(delta_in, front_entries, front_weights,entries,weights_to_update):
    '''    
    print("INPUTS: ",delta_in, front_entries, front_weights,entries,weights_to_update)    

    delta = [[[w[e]*(front_entries[e]*(1-front_entries[e]))*x for w in front_weights] for e in range(1,len(front_entries))] for x in delta_in]
    print("DELTA: ",delta)    
    delta = np.array(delta).ravel()
    '''   
    front_entries = front_entries[1:]
    front_weights = [x[1:] for x in front_weights]
    
    #[[print(delta_in[d],fw,delta_in[d]*fw) for fw in front_weights[d]] for d in range(len(delta_in))] 
    
    delta = [[delta_in[d]*fw for fw in front_weights[d]] for d in range(len(delta_in))] 
    
    delta = np.sum(delta,axis=0)
    #print(delta)
    delta = [d*fe*(1-fe) for fe,d in zip(front_entries,delta)]    
    weights_updated = [[weights_to_update[d][w]+CONT_LEARNING*(-delta[d])*entries[w] for w in range(len(weights_to_update[d]))] for d in range(len(delta))]
    
    return weights_updated,delta

def alg(entries,target):
    first_hidden_layer = [1] + activation_function(multiply_matrix(weights_input, entries))
    secnd_hidden_layer = [1] + activation_function(multiply_matrix(weights_inter, first_hidden_layer))
    #print("---",secnd_hidden_layer)
    output_layer = activation_function(multiply_matrix(weights_outpt, secnd_hidden_layer))
    
    
    if(abs(np.array(output_layer) - np.array(target)) > 0.01):
        weights_outpt_updated,delta = update_output_layer(target, output_layer,secnd_hidden_layer,weights_outpt)            
        weights_inter_updated,delta = update_hidden_layer(delta,secnd_hidden_layer,weights_outpt,first_hidden_layer,weights_inter)
        print("PRIMEIRA")
        weights_input_updated,delta = update_hidden_layer(delta,first_hidden_layer,weights_inter,entries,weights_input)
        
        print(delta)
        
    

if __name__ == "__main__":
    
    N_NEURONE_FIRST_LAYER = 2
    N_NEURONE_SECON_LAYER = 2
    N_NEURONE_OUTPT_LAYER = 1    
    
    entries = [1, 0.1, 0.6] #COPIAR TODAS AS COLUNAS MENOS A ULTIMA
    targets = [0.9] #COPIAR ULTIMA COLUNA
    
    weights_input = [[0.3, 0.2, 0.1],[0.3, 0.3, 0.2]]
    weights_inter = [[0.4, 0.4, 0.1],[0.4, 0.6, 0.3]]
    weights_outpt = [[0.5, 0.4, 0.8]]
    
    #print(targets)
    
    alg(entries, targets)
    
    #entries = [1, 0.05, 0.1]
    #targets = [0.01, 0.99]
    
    #firts_layer_weights = [[0.35, 0.15, 0.20],[0.35, 0.25, 0.30]]
    #second_layer_weights= [[0.60, 0.40, 0.45],[0.60, 0.50, 0.55]]
    
    #second_layer_outputs = [1] + activation_function(multiply_matrix(firts_layer_weights, entries))
    #outputs = trirdy_layer_outputs = activation_function(multiply_matrix(second_layer_weights, second_layer_outputs))          
    
    #second_layer_weights_updated = update_output_layer(targets, outputs,second_layer_outputs,second_layer_weights)
    #first_layer_weights_updated = update_hidden_layer(targets, outputs,second_layer_weights,second_layer_outputs,entries,firts_layer_weights)
            
        
    #print(second_layer_entries)
        
    print("FUNCAO MAIN")
    
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
    

