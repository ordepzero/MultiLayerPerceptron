# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 23:43:44 2016

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
    
def calculate_error_total(targets, outs):
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
    front_entries = front_entries[1:]
    front_weights = [x[1:] for x in front_weights]
    
    #[[print(delta_in[d],fw,delta_in[d]*fw) for fw in front_weights[d]] for d in range(len(delta_in))] 
    
    delta = [[delta_in[d]*fw for fw in front_weights[d]] for d in range(len(delta_in))] 
    
    delta = np.sum(delta,axis=0)
    #print(delta)
    delta = [d*fe*(1-fe) for fe,d in zip(front_entries,delta)]    
    weights_updated = [[weights_to_update[d][w]+CONT_LEARNING*(-delta[d])*entries[w] for w in range(len(weights_to_update[d]))] for d in range(len(delta))]
    
    return weights_updated,delta


def alg(entries,target,weights_outpt,weights_inter,weights_input):
        
    
    while True:
        first_hidden_layer = [1] + activation_function(multiply_matrix(weights_input, entries))
        secnd_hidden_layer = [1] + activation_function(multiply_matrix(weights_inter, first_hidden_layer))
        output_layer = activation_function(multiply_matrix(weights_outpt, secnd_hidden_layer))
        
        target = [0] + target
        error = pow(output_layer - target,2)/2.0
        if(error > 0.01):
            #print(error)
            weights_outpt_updated,delta = update_output_layer(target, output_layer,secnd_hidden_layer,weights_outpt)            
            weights_inter_updated,delta = update_hidden_layer(delta,secnd_hidden_layer,weights_outpt,first_hidden_layer,weights_inter)
            weights_input_updated,delta = update_hidden_layer(delta,first_hidden_layer,weights_inter,entries,weights_input)
            
            weights_outpt = weights_outpt_updated
            weights_inter = weights_inter_updated
            weights_input = weights_input_updated
        else:
            break
    return error,weights_outpt,weights_inter,weights_input #TEM QUE FICAR FORA DO WHILE
        
if __name__ == "__main__":
    
    N_NEURONE_FIRST_LAYER = 100
    N_NEURONE_SECON_LAYER = 100
    N_NEURONE_OUTPT_LAYER = 1
    
    filename = "seeds.txt"

    data = normalize_data(put_file_int_array(filename),False)
    
    
    entries = data[:,:-1] #COPIAR TODAS AS COLUNAS MENOS A ULTIMA
    targets = data[:,-1] #COPIAR ULTIMA COLUNA
    entries = [np.append(1, x)  for x in entries]  
    
    weights_input = (np.random.random((N_NEURONE_FIRST_LAYER,len(entries[0]))) - 0.5) * 2
    weights_inter = (np.random.random((N_NEURONE_SECON_LAYER,1 + N_NEURONE_FIRST_LAYER)) - 0.5) * 2
    weights_outpt = (np.random.random((N_NEURONE_OUTPT_LAYER,1 + N_NEURONE_SECON_LAYER)) - 0.5) * 2
    
        
    
    print(weights_input)    
    
    #print(data)
    '''
    #print(targets)
    epoch = 0
    while True:
        print("Epoca: ",epoch)
        error_total = 0
        for cont in range(len(entries)):
            print("Amostra: ",cont)
            error,weights_outpt,weights_inter,weights_input = alg(entries[cont], targets[cont],weights_outpt,weights_inter,weights_input)
            error_total = error_total + error
            
        error_total_m = error_total / len(entries)
        epoch = epoch + 1
        if(error_total < 0.001):
            break
    '''       
    print("FUNCAO MAIN")
    

