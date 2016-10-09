# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 10:37:02 2016

@author: PeDeNRiQue
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 04:37:09 2016

@author: PeDeNRiQue
"""
from sklearn.cluster import KMeans

import math
import numpy as np



def basis_function(center,variation, entries):
    result = sum(center-entries)
    
    return (result/(2* math.pow(variation,2)))
    #return "oi"
    

def mean_sq_dist(center,entries):
    result = sum([math.pow(d-k,2) for d,k in zip(data[0][:-1],kmeans.cluster_centers_[0])])
    return result

def put_file_int_array(filename):
    array = []
    
    with open(filename,"r") as f:
        content = f.readlines()
        for line in content: # read rest of lines
            array.append([float(x) for x in line.split()])   
    return np.array(array);
    
    
def read_file(filename):
    array = []
    
    with open(filename,"r") as f:
        content = f.readlines()
        for line in content: # read rest of lines
            array.append([x for x in line.split(",")])   
    return np.array(array);
    
def change_class_name(data,dic):
    for x in range(len(data)):
        data[x][-1] = dic[data[x][-1]]
    return data
    
def str_to_number(data):
    return[[float(j) for j in i] for i in data]
    
def order_data(data):
    n_class = [1.,2.,3.] #VERSAO ORIGINAL
    #n_class = [0.,1.,2.]
    data_alternated = []
    
    P_TRAIN = 0.75
    size_total = len(data)
    size_train = int(size_total*P_TRAIN)
    size_each_class = int(size_train / len(n_class))
    
    #print(size_train,size_each_class)
    
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
def normalize_data(f,has_target=True):
    
    x = np.array(f)
    x_normed = (x - x.min(axis=0))/ (x.max(axis=0) - x.min(axis=0))
    
    #SUBSTITUIO OS VALORES ALVO DA ULTIMA COLUNA NOS DADOS NORMALIZADOS
    if(has_target):
        x_normed[:,-1] = f[:,-1]

    return x_normed
    
def train_test_data(data,p_train=0.75):
    size_total = len(data)
    size_train = int(size_total*p_train)
    train = data[0:size_train]
    test  = data[size_train:]
    
    return train,test
    

if __name__ == '__main__':
    #data[:,-1] mostra ultima coluna
    #data[:,:-1] so não mostra ultima coluna
    filename = "wine.data"
    
    file_temp = read_file(filename)
    file_t = file_temp = str_to_number(file_temp)
    
    
    file = [[0 for i in j] for j in file_temp]
    #print(file)
    for i in range(len(file_temp)):
        for j in range(1,len(file_temp[i])):
            file[i][j-1] = file_temp[i][j]
            
    for i in range(len(file_temp)):
        file[i][-1] = file_temp[i][0]
    
    #print(file) 
    
    file_array = np.array(file)
    data = normalize_data(file_array)
    data = order_data(data)
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
    print("DATA",data[0][:-1]-centers[0],sum(data[0][:-1]-centers[0]))
    
    result = [ [basis_function(c,d,v) for c,v in zip(centers,variations)] for d in data[:,:-1]]
    
    
    print("Larguras:",result)
    
    
    
    
    
    
    
    
    
    
    
    
    