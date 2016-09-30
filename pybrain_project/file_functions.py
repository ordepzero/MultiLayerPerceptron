# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 19:35:38 2016

@author: PeDeNRiQue
"""

import numpy as np

def put_file_int_array(filename):
    array = []
    
    with open(filename,"r") as f:
        content = f.readlines()
        for line in content: # read rest of lines
            array.append([float(x) for x in line.split()])   
    return np.array(array);