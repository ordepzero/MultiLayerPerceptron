# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:19:34 2016

@author: PeDeNRiQue
"""

import math
import numpy as np

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.supervised.trainers import BackpropTrainer

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import SigmoidLayer
from pybrain.structure import FullConnection


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
  

train_data_temp,test_data_temp = train_test_data(data)

train_data = ClassificationDataSet(13, nb_classes=3)
test_data  = ClassificationDataSet(13, nb_classes=3)

cont = 0
for n in range(0, len(train_data_temp)):
    train_data.addSample( train_data_temp[n][:-1], [train_data_temp[n][-1]-1])
    #print(train_data.getSample(cont))
    #cont = cont + 1

for n in range(0, len(test_data_temp)):
    test_data.addSample( test_data_temp[n][:-1], [test_data_temp[n][-1]-1])


train_data._convertToOneOfMany( )
test_data._convertToOneOfMany( )  

print ("Number of training patterns: ", len(train_data))
print ("Input and output dimensions: ", train_data.indim, train_data.outdim)
print ("First sample (input, target, class):")
print (test_data['input'][0], test_data['target'][0], test_data['class'][0])


network = FeedForwardNetwork()

inLayer = SigmoidLayer(train_data.indim)
first_hiddenLayer = SigmoidLayer(50)
outLayer = SigmoidLayer(train_data.outdim)

network.addInputModule(inLayer)
network.addModule(first_hiddenLayer)
network.addOutputModule(outLayer)

in_to_hidden = FullConnection(inLayer, first_hiddenLayer)
hidden_to_out = FullConnection(first_hiddenLayer, outLayer)

network.addConnection(in_to_hidden)
network.addConnection(hidden_to_out)

network.sortModules()

trainer = BackpropTrainer( network, dataset=train_data, momentum=0.01, verbose=True, weightdecay=0.25)
l_error = trainer.train()
v_error = 0;
cont = 0
for i in range(100):
    error = trainer.train()
    if(abs(error - l_error) < 0.01 ):
        cont = cont + 1
        if(cont == 5):
            trnresult = percentError( trainer.testOnClassData(),
                              train_data['class'] )
            tstresult = percentError( trainer.testOnClassData(
                   dataset=test_data ), test_data['class'] )
        
            print ("epoch: %4d" % trainer.totalepochs, \
                  "  train error: %5.2f%%" % trnresult, \
                  "  test error: %5.2f%%" % tstresult)
            break
    else:
        cont = 0
    l_error = trainer.train()  
    trnresult = percentError( trainer.testOnClassData(),
                              train_data['class'] )
    tstresult = percentError( trainer.testOnClassData(
           dataset=test_data ), test_data['class'] )

    print ("epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult)
        






