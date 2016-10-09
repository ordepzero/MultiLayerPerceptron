# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 04:37:09 2016

@author: PeDeNRiQue
"""
from sklearn.cluster import KMeans

import math
import numpy as np
import file_functions as file_f
import pre_processing as pre_proc

def calculate_error(target,output):
   r = [ math.pow(aa-bb,2) for aa,bb in zip(target,output)]

   return (sum(r)/2) 
   
def calculate_output(weights,parcela):
    r = [x * y for x,y, in zip(weights,parcela)]
    return sum(r)

def convert_to_bin(value):
    if(value == 0.):
        return [0.,0.,1.]
    elif(value == 1.):
        return [0.,1.,0.]
    else:
        return [1.,0.,0.]

def basis_function(center,variation, entries):
    result = sum(center-entries)
    
    return (result/(2* math.pow(variation,2)))
    #return "oi"

def update_output_layer(targets, outputs,entries, weights,learning_rate):
    derivative_total_error = [-(target-out) for target,out in zip(targets, outputs)]
    derivative_log_function= [out*(1-out) for out in outputs]
    delta = [error*log for error,log in zip(derivative_total_error, derivative_log_function)]
    #print("-> ",targets," ", outputs," ",entries," ", weights)
    #print("DELTA:",delta)
    
    weights_updated = [[0 for x in range(len(weights[y]))] for y in range(len(weights))] 
    
        
    for i in range(len(delta)):
        for j in range(len(entries)):
            #print("< ", weights[i][j] , CONT_LEARNING , delta[i] , entries[j]," >")
            weights_updated[i][j] = weights[i][j] + learning_rate * (-delta[i] * entries[j])
            
            
    return weights_updated

def mean_sq_dist(center,entries):
    result = sum([math.pow(d-k,2) for d,k in zip(data[0][:-1],kmeans.cluster_centers_[0])])
    return result

if __name__ == '__main__':
    #data[:,-1] mostra ultima coluna
    #data[:,:-1] so não mostra ultima coluna
    filename = "iris.txt"
    dic = {'Iris-setosa\n': 0, 'Iris-versicolor\n': 1, 'Iris-virginica\n': '2'}
    
    file = file_f.read_file(filename)
    file = file_f.change_class_name(file,dic)
    file = file_f.str_to_number(file)
    file_array = np.array(file)
    data = pre_proc.normalize_data(file_array)
    
    data = file_f.order_data(data)
    n_classes = 3
    kmeans = KMeans(n_clusters = n_classes, random_state=0).fit(data[:,:-1])
    
    centers = kmeans.cluster_centers_
    
    results = [[0 for x in range(2)] for y in range(n_classes)]
         
    for x in range(len(data)):
        position = kmeans.predict([data[x][:-1]])
        results[position[0]][0] = results[position[0]][0] + 1
        MSD = mean_sq_dist(centers[position[0]],data[x][:-1])
        results[position[0]][1] = results[position[0]][1] + MSD  
        #print(results)
       
    variations = [0 for y in range(n_classes)]
    for x in range(len(results)):
        variations[x] = results[x][1] / results[x][0]
    
    
    #pseudo_samples = [print(d) for d in data[:,:-1]]
    #print(pseudo_samples)
    
    print(centers)
    print(variations)
    
    weight_output_layer = [[0.5 for j in range(n_classes)] for i in range(n_classes)]
    learning_rate = 0.25
    required_acuracy = 0.01
    
    pseudo_samples = [ [basis_function(c,v,d) for c,v in zip(centers,variations)] for d in data[:,:-1]]
    #print(weight_output_layer)
    
    targets = [convert_to_bin(d[-1]) for d in data]
    
    #print(targets)    
    
    epoch = 0
    cont = 0
    error = 0

    while True:    
        if(cont == 5):
            break
        last_error = error
        cont = cont + 1
        
        outputs = [ [calculate_output(weights,ps)for weights in weight_output_layer] for ps in pseudo_samples]
        
        
        error = [ calculate_error(target,output) for target,output in zip(targets,outputs)]
        print(error,len(error),sum(error),sum(error)/len(error))
        
        weight_output_layer = update_output_layer(targets,outputs,pseudo_samples,weight_output_layer,learning_rate)        
        
        current_error = error
        epoch = epoch + 1
    
        if(abs(current_error - last_error) <= 0.01):
            break
    
    
    
    
    
    
    
    
    
    
    