# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 16:31:36 2016

@author: PeDeNRiQue
"""

from sklearn.cluster import KMeans
import numpy as np
import math
import file_functions as file_f

def logistic(value):
    #print("_",value)
    return 1/(1+math.exp(-value))
    
def mean_sq_dist(center,entries):
    result = sum([math.pow(d-k,2) for d,k in zip(center,entries)])
    return result

def radial_basis_function(x, w, variation):
    dif =[math.pow(xx - ww,2)  for xx,ww in zip(x,w)]
    result = sum(dif)
    return math.exp(-result/(2*variation))
    
def update_weights(entries,weights,deltas,learn_r):
    #print("PARAMS:",entries,weights,deltas,learn_r)
    updated_weights = [weight+(learn_r*delta*entries) for weight,delta in zip(weights,deltas)]    
    #print("UPDATED:",updated_weights)    
    return updated_weights
    
def calculate_error(outputs, desireds):
    error = [ (math.pow(o - d,2))/2 for o,d in zip(outputs, desireds)]
    return sum(error)/len(desireds)
    
if __name__ == '__main__':
    
    filename = "samples.txt"
    
    n_outputs = 1
    n_classes = 2
    learning_rate = 0.5
    
    output_layer_weights = np.random.random((n_outputs, n_classes+1))    
    
    file = file_f.read_file(filename," ")
    data = np.array(file_f.str_to_number(file))
    desired_output = np.array([[d] for d in data[:,-1]])
    

    kmeans = KMeans(n_clusters = n_classes, random_state=0).fit(data[:,:-1])

    #CONTAR O NÚMERO DE EXEMPLOS EM CADA CLUSTER    
    #SOMAR O QUADRADO DA DIFERENCA DE CADA AMOSTRA E O CENTRO DO SEU GRUPO 
    results = [[0 for x in range(2)] for y in range(n_classes)]       
    
    #DETERMINANDO OS CENTROS
    centers = kmeans.cluster_centers_
    
    #CALCULANDO A VARIANCIA DE CADA UMA DAS FUNÇÕES
    for x in range(len(data)):
        positions = kmeans.predict([data[x][:-1]])
        position = positions[0]
        results[position][0] = results[position][0] + 1
        MSD = mean_sq_dist(centers[position],data[x][:-1])
        results[position][1] = results[position][1] + MSD 
        
    #CALCULO DA VARIANCIA DE CADA FUNÇÃO
    variations = [0 for y in range(n_classes)]
    for x in range(len(results)):
        variations[x] = results[x][1] / results[x][0]
    
    #print(results)
    #print(variations)
    
    
    #CALCULANDO AS ENTRADAS DA CAMADA DE SAIDA
    pseudo_samples = [np.append(-1,[radial_basis_function(d[:-1],centers[c],variations[c]) for c in range(len(centers))]) for d in data]
            
    
    #print(pseudo_samples)
    epoch = 0 
    current_error = 0
    last_error = 0
    result = 0
    #sample_error = 0
    while True:
        
        for ps in range(len(pseudo_samples)):
            
            while True:
                sample_error = 0 
                outputs = [logistic(sum(pseudo_samples[ps]*weight)) for weight in output_layer_weights]
                deltas = (desired_output[ps]-outputs)*(outputs*(np.array([1])-outputs))
                #print(outputs,desired_output[ps],deltas)
                            
                #CALCULANDO O ERRO
                sample_error = calculate_error(outputs,desired_output[ps])
                if(sample_error > 0.1):
                    #print(outputs)
                    result = update_weights(pseudo_samples[ps],output_layer_weights,deltas,learning_rate)
                    #print(result)
                    #output_layer_weights = result
                    for x in range(len(result)):
                        n = [[n for n in nw] for nw in result]
                    #print("OUT",outputs)
                    #print("N",n)
                    
                    output_layer_weights = n
                else:
                    break
                print(sample_error)
        epoch = epoch + 1
        if(epoch == 1):
            break
    
    #FALTA TERMINAR TRINAMENTO
    
    
    
    
    
    print(output_layer_weights,result)
    
    
    
    
    
    
    
    