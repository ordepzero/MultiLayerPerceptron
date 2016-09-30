# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 23:43:44 2016

@author: PeDeNRiQue
"""

import math
import numpy as np


CONT_LEARNING = 0.25
P_TRAIN = 0.7
ALFA = 0.5

weights_input = []#MATRIZ DE PESOS DA CAMADA DE ENTRADA
weights_inter = []#MATRIZ DE PESOS DA CAMADA INTERMEDIATIA
weights_outpt = []#MATRIZ DE PESOS DA CAMADA DE SAIDA

N_NEURONE_FIRST_LAYER = 150
N_NEURONE_SECON_LAYER = 150
N_NEURONE_OUTPT_LAYER = 1

file_results = open('results_teste1.txt', 'w')

def inicialize_matrix_zeros(rows, columns):
    grade = [0] * rows
    for i in range(rows):
        grade[i] = [0] * columns 
        
    return np.array(grade)

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
    #print("_",value)
    return 1/(1+math.exp(-value))
    
def multiply_matrix(m, c):
    m = np.array(m)
    c = np.array(c)
    r = m * c
    result = np.sum(r,axis=1).tolist()
    #print("<",len(result),">")
    return (result)
    
def activation_function(values):
    values = [ logistic(v) for v in values]
    #print(values)
    return values
    
def calculate_error_total(targets, outs):
    result =  [pow((target - out),2)/2 for target,out in zip(targets, outs)]
    return sum(result)   
 
def update_output_layer(targets, outputs,entries, weights, weights_m,act_pots):  
    #print(outputs,act_pots)
    derivative_total_error = [-(target-out) for target,out in zip(targets, outputs)]
    derivative_log_function= [act_pot*(1-act_pot) for act_pot in outputs]
    delta = [error*log for error,log in zip(derivative_total_error, derivative_log_function)]
    
    weights_updated = [[0 for x in range(len(weights[y]))] for y in range(len(weights))] 
    
        
    for i in range(len(delta)):
        for j in range(len(entries)):
            #print("< ", weights[i][j] , CONT_LEARNING , delta[i] , entries[j]," >")
            weights_updated[i][j] = weights[i][j] + CONT_LEARNING * (-delta[i] * entries[j]) + (ALFA * weights_m[i][j])
            
            
    return weights_updated,delta


def update_hidden_layer(delta_in, front_entries, front_weights,entries,weights_to_update, weights_m,act_pots):
    
    #print(front_entries)    
    
    front_entries = front_entries[1:]
    front_weights = [x[1:] for x in front_weights]
    
    #[[print(delta_in[d],fw,delta_in[d]*fw) for fw in front_weights[d]] for d in range(len(delta_in))] 
    
    delta = [[delta_in[d]*fw for fw in front_weights[d]] for d in range(len(delta_in))] 
    
    delta = np.sum(delta,axis=0)
    #print(delta)
    delta = [d*fe*(1-fe) for fe,d in zip(front_entries,delta)]    
    weights_updated = [[weights_to_update[d][w]+CONT_LEARNING*(-delta[d])*entries[w] + (ALFA * weights_m[d][w]) for w in range(len(weights_to_update[d]))] for d in range(len(delta))]
    
    return weights_updated,delta


def alg(entries,target,weights_outpt,weights_inter,weights_input,weights_outpt_m,weights_inter_m,weights_input_m):
        
    
    while True:
        first_hidden_layer = [1] + activation_function(multiply_matrix(weights_input, entries))
        secnd_hidden_layer = [1] + activation_function(multiply_matrix(weights_inter, first_hidden_layer))
        output_layer = activation_function(multiply_matrix(weights_outpt, secnd_hidden_layer))
        
        target = [0] + target
        error = pow(output_layer - target,2)/2.0
        if(error > 0.1):
            #print(error)
            weights_outpt_updated,delta = update_output_layer(target, output_layer,secnd_hidden_layer,weights_outpt,weights_outpt_m)            
            weights_inter_updated,delta = update_hidden_layer(delta,secnd_hidden_layer,weights_outpt,first_hidden_layer,weights_inter,weights_inter_m)
            weights_input_updated,delta = update_hidden_layer(delta,first_hidden_layer,weights_inter,entries,weights_input,weights_input_m)
            
            weights_outpt_updatedt = np.array(weights_outpt_updated)
            weights_inter_updatedt = np.array(weights_inter_updated)
            weights_input_updatedt = np.array(weights_input_updated)
            
            weights_outpt_m = weights_outpt_updatedt - weights_outpt
            weights_inter_m = weights_inter_updatedt - weights_inter
            weights_input_m = weights_input_updatedt - weights_input
            
            weights_outpt = weights_outpt_updated
            weights_inter = weights_inter_updated
            weights_input = weights_input_updated
        else:
            break
    return error,weights_outpt,weights_inter,weights_input,weights_outpt_m,weights_inter_m,weights_input_m #TEM QUE FICAR FORA DO WHILE


    
def trainning(data,n_first_layer,n_secon_layer,n_output_layer):
    entries = data[:,:-1] #COPIAR TODAS AS COLUNAS MENOS A ULTIMA
    targets = data[:,-1] #COPIAR ULTIMA COLUNA
    entries = [np.append(1, x)  for x in entries]  
    
    
    weights_input = (np.random.random((n_first_layer,len(entries[0]))) - 0.5) * 2
    weights_inter = (np.random.random((n_secon_layer,1 + n_first_layer)) - 0.5) * 2
    weights_outpt = (np.random.random((n_output_layer,1 + n_secon_layer)) - 0.5) * 2
    
    #WEOGHTS TO MOMENTUM
    weights_input_m = inicialize_matrix_zeros(n_first_layer,len(entries[0]))
    weights_inter_m = inicialize_matrix_zeros(n_secon_layer,1 + n_first_layer)
    weights_outpt_m = inicialize_matrix_zeros(n_output_layer,1 + n_secon_layer)
    
    epoch = 0
    last_error = 0
    while True:
        
        error_total = 0
        for cont in range(len(entries)):
            #print("Amostra: ",cont)
            error,weights_outpt,weights_inter,weights_input,weights_outpt_m,weights_inter_m,weights_input_m = alg(entries[cont], targets[cont],weights_outpt,weights_inter,weights_input,weights_outpt_m,weights_inter_m,weights_input_m)
            error_total = error_total + error
            
        error_total_m = error_total / len(entries)
        epoch = epoch + 1
        #print("ERROR TOTAL MEDIO: ",error_total_m,last_error)
        if(error_total_m < 0.01 or last_error == error_total_m):
            file_results.write("Ciclos: "+str((epoch)))
            print("Epoca: ",epoch)
            return weights_outpt,weights_inter,weights_input
        last_error = error_total_m
    file_results.write("Ciclos: "+str((epoch)))
    return weights_outpt,weights_inter,weights_input



def calculate_output(entries,weights_outpt,weights_inter,weights_input):
    first_hidden_layer = [1] + activation_function(multiply_matrix(weights_input, entries))
    secnd_hidden_layer = [1] + activation_function(multiply_matrix(weights_inter, first_hidden_layer))
    output_layer = activation_function(multiply_matrix(weights_outpt, secnd_hidden_layer))

    return output_layer
        
    
def test_net(data,weights_outpt,weights_inter,weights_input):
    data = np.array(data)

    entries = data[:,:-1] #COPIAR TODAS AS COLUNAS MENOS A ULTIMA
    targets = data[:,-1] #COPIAR ULTIMA COLUNA
    entries = [np.append(1, x)  for x in entries]
    cont = 0
    error_total = 0
    for entry,target in zip(entries,targets):
        cont = cont + 1    
        result = calculate_output(entry,weights_outpt,weights_inter,weights_input)
        error = pow(target - result,2)/2
        error_total = error_total + error
        #print(cont,result,target,error)
        
    file_results.write("ERRO QUADRADO MEDIO: "+str((error_total[0]/cont)))
    file_results.write("\n")     
    print("ERRO QUADRATICO MEDIO:",error_total/cont)
        
    return "FIM"
    

def two_layers():
    N_NEURONE_FIRST_LAYER = 150
    N_NEURONE_SECON_LAYER = 150
    N_NEURONE_OUTPT_LAYER = 1
    
    filename = "teste1.txt"

    data = normalize_data(put_file_int_array(filename),False)
    
    P_TRAIN = 0.9
    size_total = len(data)
    size_train = int(size_total*P_TRAIN)
    train = data[0:size_train]
    test  = data[size_train:]
    
    print("Numero de neuronios:",N_NEURONE_FIRST_LAYER)   
    print("Tamanho da base:",P_TRAIN," (%) ,",size_train)
    print("APrendizagem:",CONT_LEARNING)
    print("Momentum:",ALFA)      
    
    #ARMAZENANDO RESULTADOS
    file_results.write("Nome do arquivo: "+filename)
    file_results.write("\n")
    file_results.write(str((str(N_NEURONE_FIRST_LAYER)+" "+str(N_NEURONE_SECON_LAYER)+" "+str(N_NEURONE_OUTPT_LAYER))))
    file_results.write("\n")    
    file_results.write("BASE DE TREINO (%):"+str((P_TRAIN)))    
    file_results.write("\n")    

    weights_outpt,weights_inter,weights_input = trainning(train,N_NEURONE_FIRST_LAYER,N_NEURONE_SECON_LAYER,N_NEURONE_OUTPT_LAYER)
    test_net(test,weights_outpt,weights_inter,weights_input)
    
    #file_results.close()
    print("FUNCAO MAIN")


def alg_one_layer(entries,target,weights_outpt,weights_input,weights_outpt_m,weights_input_m):
        
    
    while True:
        act_pot1 = multiply_matrix(weights_input, entries)
        first_hidden_layer = [1] + activation_function(act_pot1)
        act_pot2 = multiply_matrix(weights_outpt, first_hidden_layer)
        #print("<",weights_outpt,">")
        output_layer = activation_function(act_pot2)
        
        target = [0] + target
        error = pow(output_layer - target,2)/2.0
        if(error > 0.1):
            #print(error)
            weights_outpt_updated,delta = update_output_layer(target, output_layer,first_hidden_layer,weights_outpt,weights_outpt_m,act_pot2)            
            weights_input_updated,delta = update_hidden_layer(delta,first_hidden_layer,weights_outpt,entries,weights_input,weights_input_m,act_pot1)
            
            weights_outpt_updatedt = np.array(weights_outpt_updated)
            weights_input_updatedt = np.array(weights_input_updated)
            
            weights_outpt_m = weights_outpt_updatedt - weights_outpt
            weights_input_m = weights_input_updatedt - weights_input
            
            weights_outpt = weights_outpt_updated
            weights_input = weights_input_updated
        else:
            break
    return error,weights_outpt,weights_input,weights_outpt_m,weights_input_m #TEM QUE FICAR FORA DO WHILE


    
def trainning_one_layer(data,n_first_layer,n_output_layer):
    entries = data[:,:-1] #COPIAR TODAS AS COLUNAS MENOS A ULTIMA
    targets = data[:,-1] #COPIAR ULTIMA COLUNA
    entries = [np.append(1, x)  for x in entries]  
    
    
    weights_input = (np.random.random((n_first_layer,len(entries[0]))) - 0.5) * 2
    weights_outpt = (np.random.random((n_output_layer,1 + n_first_layer)) - 0.5) * 2
    
    #WEOGHTS TO MOMENTUM
    weights_input_m = inicialize_matrix_zeros(n_first_layer,len(entries[0]))
    weights_outpt_m = inicialize_matrix_zeros(n_output_layer,1 + n_first_layer)
    
    epoch = 0
    last_error = 0
    while True:
        
        error_total = 0
        for cont in range(len(entries)):
            #print("Amostra: ",cont)
            error,weights_outpt,weights_input,weights_outpt_m,weights_input_m = alg_one_layer(entries[cont], targets[cont],weights_outpt,weights_input,weights_outpt_m,weights_input_m)
            error_total = error_total + error
            
        error_total_m = error_total / len(entries)
        epoch = epoch + 1
        #print("ERROR TOTAL MEDIO: ",error_total_m,last_error)
        if(error_total_m < 0.01 or last_error == error_total_m):
            file_results.write("Ciclos: "+str((epoch)))
            print("Epoca: ",epoch)
            return weights_outpt,weights_input
        last_error = error_total_m
    file_results.write("Ciclos: "+str((epoch)))
    return weights_outpt,weights_input



def calculate_output_one_layer(entries,weights_outpt,weights_input):
    first_hidden_layer = [1] + activation_function(multiply_matrix(weights_input, entries))
    output_layer = activation_function(multiply_matrix(weights_outpt, first_hidden_layer))

    return output_layer
        
    
def test_net_one_layer(data,weights_outpt,weights_input):
    data = np.array(data)

    entries = data[:,:-1] #COPIAR TODAS AS COLUNAS MENOS A ULTIMA
    targets = data[:,-1] #COPIAR ULTIMA COLUNA
    entries = [np.append(1, x)  for x in entries]
    cont = 0
    error_total = 0
    for entry,target in zip(entries,targets):
        cont = cont + 1    
        result = calculate_output_one_layer(entry,weights_outpt,weights_input)
        error = pow(target - result,2)/2
        error_total = error_total + error
        #print(cont,result,target,error)
        
    file_results.write("ERRO QUADRADO MEDIO: "+str((error_total[0]/cont)))
    file_results.write("\n")     
    print("ERRO QUADRATICO MEDIO:",error_total/cont)
        
    return "FIM"
    

def one_layers():
    N_NEURONE_FIRST_LAYER = 3
    N_NEURONE_OUTPT_LAYER = 1
    
    filename = "teste1.txt"

    data = normalize_data(put_file_int_array(filename),False)
    
    P_TRAIN = 0.90
    size_total = len(data)
    size_train = int(size_total*P_TRAIN)
    train = data[0:size_train]
    test  = data[size_train:]
    
    #ARMAZENANDO RESULTADOS
    file_results.write("Numero de camadas: 1\n")
    file_results.write("\n")
    file_results.write("Nome do arquivo: "+filename)
    file_results.write("\n")
    file_results.write(str((str(N_NEURONE_FIRST_LAYER)+" "+" "+str(N_NEURONE_OUTPT_LAYER))))
    file_results.write("\n")    
    file_results.write("BASE DE TREINO (%):"+str((P_TRAIN)))    
    file_results.write("\n")    

    weights_outpt,weights_input = trainning_one_layer(train,N_NEURONE_FIRST_LAYER,N_NEURONE_OUTPT_LAYER)
    test_net_one_layer(test,weights_outpt,weights_input)
    
    #file_results.close()
    print("FUNCAO MAIN")
    
if __name__ == "__main__":
    
    for x in range(5):
        if(False):
            print("Numero de camadas intermediarias: 2")
            two_layers()
        else:
            print("Numero de camadas intermediarias: 1")
            one_layers()
        print("\n\n")
    file_results.close()
    
''' 
#IMPLEMENTAÇÃO DO X_FOLD

folds = 10
parts = np.array_split(data, folds)

for fold in range(folds):
    train = []
    test  = []
    
    for f in range(folds):
        if(f == fold):
            test = parts[f]
        else:
            if(len(train) == 0):
                train = parts[f]
            else:
                train = np.append(train,parts[f], axis=0)

    
    weights_outpt,weights_inter,weights_input = trainning(train)
    test_net(test,weights_outpt,weights_inter,weights_input)
'''
