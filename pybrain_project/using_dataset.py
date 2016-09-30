# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 23:29:04 2016

@author: PeDeNRiQue
"""

import file_functions as file_f
import pre_processing as pre_proc
import numpy as np

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import SigmoidLayer
from pybrain.structure import FullConnection


filename = "seeds.txt"
data = pre_proc.normalize_data(file_f.put_file_int_array(filename))    
train_data_temp,test_data_temp = pre_proc.train_test_data(data)

train_data = ClassificationDataSet(7, nb_classes=3)
test_data  = ClassificationDataSet(7, nb_classes=3)

for n in range(0, len(train_data_temp)):
    train_data.addSample( train_data_temp[n][:-1], [train_data_temp[n][-1]-1])
    print(train_data.getSample(n))

for n in range(0, len(test_data_temp)):
    test_data.addSample( test_data_temp[n][:-1], [test_data_temp[n][-1]-1])


train_data._convertToOneOfMany( )
test_data._convertToOneOfMany( )  

print ("Number of training patterns: ", len(train_data))
print ("Input and output dimensions: ", train_data.indim, train_data.outdim)
print ("First sample (input, target, class):")
print (train_data['input'][0], train_data['target'][0], train_data['class'][0])


network = FeedForwardNetwork()

inLayer = SigmoidLayer(train_data.indim)
hiddenLayer = SigmoidLayer(3)
outLayer = SigmoidLayer(train_data.outdim)

network.addInputModule(inLayer)
network.addModule(hiddenLayer)
network.addOutputModule(outLayer)

in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

network.addConnection(in_to_hidden)
network.addConnection(hidden_to_out)

network.sortModules()

trainer = BackpropTrainer( network, dataset=train_data, momentum=0.1, verbose=True, weightdecay=0.01)

for i in range(20):
    trainer.trainEpochs( 1 )
    trnresult = percentError( trainer.testOnClassData(),
                              train_data['class'] )
    tstresult = percentError( trainer.testOnClassData(
           dataset=test_data ), test_data['class'] )

    print ("epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult)

















