# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 12:11:13 2016

@author: PeDeNRiQue
"""

import math
import numpy as np



a = np.array([ 0.3 , 0.2])
b = np.array([ 0.3 ,  0.78]) 
v = 0.04560000000000001

r = [math.pow(aa - bb,2) for aa,bb in zip(a,b)]

print(sum(r)/(2*v))