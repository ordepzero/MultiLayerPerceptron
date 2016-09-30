# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 22:32:38 2016

@author: PeDeNRiQue
"""
import file_functions as file_f
import pre_processing as pre_proc

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
data = pre_proc.normalize_data(file_f.put_file_int_array(filename),False)    
train_data,test_data = pre_proc.train_test_data(data)

#trndata._convertToOneOfMany( )
#tstdata._convertToOneOfMany( )  

print ("Number of training patterns: ", len(train_data))
print ("Input and output dimensions: ", train_data.indim, train_data.outdim)
print ("First sample (input, target, class):")
#print (trndata['input'][0], trndata['target'][0], trndata['class'][0])

#fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )

#PYBRAIN
network = FeedForwardNetwork()

inLayer = SigmoidLayer(7)
hiddenLayer = SigmoidLayer(3)
outLayer = SigmoidLayer(3)

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
                              train_data[:,-1] )
    tstresult = percentError( trainer.testOnClassData(
           dataset=test_data ), test_data[:,-1] )

    print ("epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult)
'''    
    out = network.activateOnDataset(griddata)
    out = out.argmax(axis=1)  # the highest output activation gives the class
    out = out.reshape(X.shape)
    
    figure(1)
    ioff()  # interactive graphics off
    clf()   # clear the plot
    hold(True) # overplot on
    for c in [0,1,2]:
        here, _ = where(tstdata['class']==c)
        plot(tstdata['input'][here,0],tstdata['input'][here,1],'o')
    if out.max()!=out.min():  # safety check against flat field
        contourf(X, Y, out)   # plot the contour
    ion()   # interactive graphics on
    draw()  # update the plot    

ioff()
show()

'''