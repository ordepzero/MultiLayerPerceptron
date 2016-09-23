# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 23:43:44 2016

@author: PeDeNRiQue
"""

import math
import numpy as np


CONT_LEARNING = 0.25
P_TRAIN = 0.75
ALFA = 0.1

weights_input = []#MATRIZ DE PESOS DA CAMADA DE ENTRADA
weights_inter = []#MATRIZ DE PESOS DA CAMADA INTERMEDIATIA
weights_outpt = []#MATRIZ DE PESOS DA CAMADA DE SAIDA

N_NEURONE_FIRST_LAYER = 150
N_NEURONE_SECON_LAYER = 150
N_NEURONE_OUTPT_LAYER = 1

file_results = open('results_seeds.txt', 'w')

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
 
def convert_to_bin_class(value):
    if(value == 1.):
        return [0, 0, 1]
    elif(value == 2.):
        return [0, 1, 0]
    else:
        return [1, 0, 0]
        
def convert_to_list_int(values):
    index = 0
    new_values = [0] * len(values)
    for i in range(len(values)):
        if(values[i] > values[index]):
            index = i
    new_values[index] = 1
    
    return new_values
    
def update_output_layer(targets, outputs,entries, weights, weights_m):    
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
            weights_updated[i][j] = weights[i][j] + CONT_LEARNING * (-delta[i] * entries[j]) + (ALFA * weights_m[i][j])
            
            
    return weights_updated,delta


def update_hidden_layer(delta_in, front_entries, front_weights,entries,weights_to_update, weights_m):
    
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
        
        error = [pow((t - out),2)/2 for t,out in zip(target, output_layer)]
        mean_error = sum(error)/len(error)
        #
        if(mean_error > 0.1):
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
    return mean_error,weights_outpt,weights_inter,weights_input,weights_outpt_m,weights_inter_m,weights_input_m #TEM QUE FICAR FORA DO WHILE


    
def trainning(data):
    entries = data[:,:-1] #COPIAR TODAS AS COLUNAS MENOS A ULTIMA
    targets = data[:,-1] #COPIAR ULTIMA COLUNA
    entries = [np.append(1, x)  for x in entries]  
    
    targets = [convert_to_bin_class(t) for t in targets]
    
    weights_input = (np.random.random((N_NEURONE_FIRST_LAYER,len(entries[0]))) - 0.5) * 2
    weights_inter = (np.random.random((N_NEURONE_SECON_LAYER,1 + N_NEURONE_FIRST_LAYER)) - 0.5) * 2
    weights_outpt = (np.random.random((N_NEURONE_OUTPT_LAYER,1 + N_NEURONE_SECON_LAYER)) - 0.5) * 2
    
    #WEOGHTS TO MOMENTUM
    weights_input_m = inicialize_matrix_zeros(N_NEURONE_FIRST_LAYER,len(entries[0]))
    weights_inter_m = inicialize_matrix_zeros(N_NEURONE_SECON_LAYER,1 + N_NEURONE_FIRST_LAYER)
    weights_outpt_m = inicialize_matrix_zeros(N_NEURONE_OUTPT_LAYER,1 + N_NEURONE_SECON_LAYER)
    
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
    targets = [convert_to_bin_class(t) for t in targets]
    cont = 0
    
    for entry,target in zip(entries,targets):
        result = calculate_output(entry,weights_outpt,weights_inter,weights_input)
        
        result = convert_to_list_int(result)
        if(result == target):
            cont = cont + 1
        
    print(cont,len(data))
    file_results.write("ACURACIA: "+str((cont/len(data))))
    file_results.write("\n")
        
    return "FIM"
    

def two_layers():
    global N_NEURONE_FIRST_LAYER
    global N_NEURONE_SECON_LAYER  
    global N_NEURONE_OUTPT_LAYER
    
    N_NEURONE_FIRST_LAYER = 5
    N_NEURONE_SECON_LAYER = 2
    N_NEURONE_OUTPT_LAYER = 3
    
    filename = "seeds.txt"

    data = load_data(filename)
    
    P_TRAIN = 0.75
    size_total = len(data)
    size_train = int(size_total*P_TRAIN)
    train = data[0:size_train]
    test  = data[size_train:]
    
    #ARMAZENANDO RESULTADOS
    file_results.write("Nome do arquivo: "+filename)
    file_results.write("\n")
    file_results.write(str((str(N_NEURONE_FIRST_LAYER)+" "+str(N_NEURONE_SECON_LAYER)+" "+str(N_NEURONE_OUTPT_LAYER))))
    file_results.write("\n")    
    file_results.write("BASE DE TREINO (%):"+str((P_TRAIN)))    
    file_results.write("\n")    

    weights_outpt,weights_inter,weights_input = trainning(train)
    test_net(test,weights_outpt,weights_inter,weights_input)
    
    file_results.close()
    print("FUNCAO MAIN")


def alg_one_layer(entries,target,weights_outpt,weights_input,weights_outpt_m,weights_input_m):
        
    
    while True:
        first_hidden_layer = [1] + activation_function(multiply_matrix(weights_input, entries))
        output_layer = activation_function(multiply_matrix(weights_outpt, first_hidden_layer))
        
        error = [pow((t - out),2)/2 for t,out in zip(target, output_layer)]
        mean_error = sum(error)/len(error)
        #
        if(mean_error > 0.1):
            #print(error)
            weights_outpt_updated,delta = update_output_layer(target, output_layer,first_hidden_layer,weights_outpt,weights_outpt_m)            
            weights_input_updated,delta = update_hidden_layer(delta,first_hidden_layer,weights_outpt,entries,weights_input,weights_input_m)
            
            weights_outpt_updatedt = np.array(weights_outpt_updated)
            weights_input_updatedt = np.array(weights_input_updated)
            
            weights_outpt_m = weights_outpt_updatedt - weights_outpt
            weights_input_m = weights_input_updatedt - weights_input
            
            weights_outpt = weights_outpt_updated
            weights_input = weights_input_updated
        else:
            break
    return mean_error,weights_outpt,weights_input,weights_outpt_m,weights_input_m #TEM QUE FICAR FORA DO WHILE


    
def trainning_one_layer(data):
    entries = data[:,:-1] #COPIAR TODAS AS COLUNAS MENOS A ULTIMA
    targets = data[:,-1] #COPIAR ULTIMA COLUNA
    entries = [np.append(1, x)  for x in entries]  
    
    targets = [convert_to_bin_class(t) for t in targets]
    
    weights_input = (np.random.random((N_NEURONE_FIRST_LAYER,len(entries[0]))) - 0.5) * 2
    weights_outpt = (np.random.random((N_NEURONE_OUTPT_LAYER,1 + N_NEURONE_FIRST_LAYER)) - 0.5) * 2
    
    #WEOGHTS TO MOMENTUM
    weights_input_m = inicialize_matrix_zeros(N_NEURONE_FIRST_LAYER,len(entries[0]))
    weights_outpt_m = inicialize_matrix_zeros(N_NEURONE_OUTPT_LAYER,1 + N_NEURONE_FIRST_LAYER)
    
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
    targets = [convert_to_bin_class(t) for t in targets]
    cont = 0
    
    for entry,target in zip(entries,targets): 
        result = calculate_output_one_layer(entry,weights_outpt,weights_input)
        result = convert_to_list_int(result)
        if(result == target):
            cont = cont + 1
        
    print(cont,len(data))
    file_results.write("ACURACIA: "+str((cont/len(data))))
    file_results.write("\n")     
        
    return "FIM"
    
def class_index(values,value):
    for i in range(len(values)):
        if(values[i] == value):
            return i;

def load_data(filename):
    n_class = [1.,2.,3.]
    data_alternated = []
    
    data = normalize_data(put_file_int_array(filename))
    
    P_TRAIN = 0.75
    size_total = len(data)
    size_train = int(size_total*P_TRAIN)
    size_each_class = int(size_train / len(n_class))
    
    print(size_train,size_each_class)
    
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
    
def one_layer():
    global N_NEURONE_FIRST_LAYER
    global N_NEURONE_OUTPT_LAYER    
    
    N_NEURONE_FIRST_LAYER = 5
    N_NEURONE_OUTPT_LAYER = 3
    
    filename = "seeds.txt"
    
    data = load_data(filename)
    
    P_TRAIN = 0.75
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

    weights_outpt,weights_input = trainning_one_layer(train)
    test_net_one_layer(test,weights_outpt,weights_input)
    
    file_results.close()
    print("FUNCAO MAIN")

 
    
if __name__ == "__main__":
    two_layers()
    
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
