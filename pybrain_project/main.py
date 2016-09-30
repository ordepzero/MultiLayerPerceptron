# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 19:34:28 2016

@author: PeDeNRiQue
"""

import file_functions as file_f
import pre_processing as pre_proc

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import SigmoidLayer
from pybrain.structure import FullConnection

if __name__ == "__main__":
    
    filename = "teste1.txt"
    
    data = pre_proc.normalize_data(file_f.put_file_int_array(filename),False)
    
    train_data,test_data = pre_proc.train_test_data(data)
    print(len(test_data))
    
    #PYBRAIN
    network = FeedForwardNetwork()
    
    inLayer = SigmoidLayer(2)
    hiddenLayer = SigmoidLayer(3)
    outLayer = SigmoidLayer(1)
    
    network.addInputModule(inLayer)
    network.addModule(hiddenLayer)
    network.addOutputModule(outLayer)
    
    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)
    
    network.addConnection(in_to_hidden)
    network.addConnection(hidden_to_out)
    
    network.sortModules()
    
    
    print(type(network))
    
    
    
    
    
    
    
    
    
    