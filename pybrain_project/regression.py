# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 02:16:32 2016

@author: PeDeNRiQue
"""
import math
import file_functions as file_f
import pre_processing as pre_proc

from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import SigmoidLayer
from pybrain.structure import FullConnection


filename = "teste1.txt"
data = pre_proc.normalize_data(file_f.put_file_int_array(filename),False)    
train_data_temp,test_data_temp = pre_proc.train_test_data(data)

train_data = SupervisedDataSet(14, 1)
test_data  = SupervisedDataSet(14, 1)

cont = 0
for n in range(0, len(train_data_temp)):
    train_data.addSample( train_data_temp[n][:-1], [train_data_temp[n][-1]-1])

for n in range(0, len(test_data_temp)):
    test_data.addSample( test_data_temp[n][:-1], [test_data_temp[n][-1]-1])


#train_data._convertToOneOfMany( )
#test_data._convertToOneOfMany( )  

print ("Number of training patterns: ", len(train_data))
print ("Input and output dimensions: ", train_data.indim, train_data.outdim)
print ("First sample (input, target, class):")
print (test_data['input'][0], test_data['target'][0])


network = FeedForwardNetwork()

inLayer = SigmoidLayer(train_data.indim)
first_hiddenLayer = SigmoidLayer(100)
second_hiddenLayer= SigmoidLayer(100)
outLayer = SigmoidLayer(train_data.outdim)

network.addInputModule(inLayer)
network.addModule(first_hiddenLayer)
network.addModule(second_hiddenLayer)
network.addOutputModule(outLayer)

in_to_hidden = FullConnection(inLayer, first_hiddenLayer)
hidden_to_hidden = FullConnection(first_hiddenLayer,second_hiddenLayer)
hidden_to_out = FullConnection(second_hiddenLayer, outLayer)

network.addConnection(in_to_hidden)
network.addConnection(hidden_to_hidden)
network.addConnection(hidden_to_out)

network.sortModules()

trainer = BackpropTrainer( network, dataset=train_data, momentum=0.1, verbose=True, weightdecay=0.05)


#trainer.trainUntilConvergence(maxEpochs = 100)
for i in range(20):
    mse = trainer.train()
    rmse = math.sqrt( mse )
    print("=",mse,rmse)
    
    









