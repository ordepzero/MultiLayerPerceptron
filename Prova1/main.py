# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 08:16:11 2016

@author: PeDeNRiQue
"""
#PRIMEIRA QUESTÃO
def hard_limiter(value):
    if(value >= 0):
        return 1
    else:
        return -1

first_layer = [ [1 for i in range(4)] for x in range(4)]
bias = [0.5, 1.5, 2.5, 3.5]
pesos = [1,-1,1,-1]

for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                #print(i,j,k,l)
                result = [hard_limiter(i+j+k+l-b) for b in bias]
                #print(result)
                result2 = sum([ r*p for r,p in zip(result,pesos)])
                #print("=",result)                
                print(hard_limiter(result2))
                
                
                