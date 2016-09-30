# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 20:00:49 2016

@author: PeDeNRiQue
"""

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

means = [(-1,0),(2,4),(3,1)]
cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
alldata = ClassificationDataSet(2, 1, nb_classes=3)

for n in range(400):
    for klass in range(3):
        input = multivariate_normal(means[klass],cov[klass])
        alldata.addSample(input, [klass])

tstdata_temp, trndata_temp = alldata.splitWithProportion( 0.25 )

tstdata = ClassificationDataSet(2, 1, nb_classes=3)
for n in range(0, tstdata_temp.getLength()):
    tstdata.addSample( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )
    print(tstdata_temp.getSample(n))

trndata = ClassificationDataSet(2, 1, nb_classes=3)
for n in range(0, trndata_temp.getLength()):
    trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )

trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )  

print ("Number of training patterns: ", len(trndata))
print ("Input and output dimensions: ", trndata.indim, trndata.outdim)
print ("First sample (input, target, class):")
print (trndata['input'][0], trndata['target'][0], trndata['class'][0])



#fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )

#PYBRAIN
network = FeedForwardNetwork()

inLayer = SigmoidLayer(2)
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

trainer = BackpropTrainer( network, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)



ticks = arange(-3.,6.,0.2)
X, Y = meshgrid(ticks, ticks)
# need column vectors in dataset, not arrays
griddata = ClassificationDataSet(2,1, nb_classes=3)
for i in range(X.size):
    griddata.addSample([X.ravel()[i],Y.ravel()[i]], [0])
griddata._convertToOneOfMany()

for i in range(20):
    trainer.trainEpochs( 1 )
    trnresult = percentError( trainer.testOnClassData(),
                              trndata['class'] )
    tstresult = percentError( trainer.testOnClassData(
           dataset=tstdata ), tstdata['class'] )

    print ("epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult)
    
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

