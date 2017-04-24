# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 21:46:31 2017
@author: RandySteven
"""
import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - np.tanh(x)**2


nlayers=int(input('Digite numero de capas='))

n=[]
for i in range(0,nlayers):
    n.append(int(input('Numero de Neuronas de la capa '+str(i+1)+'=')))
    

#Observations and real values: X values to estimate, Y real values.
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T

#weight values
syn=[]
nant=3
for i in range(0,nlayers-1):
    syn.append(2*np.random.random((nant,n[i])) - 1)
    nant=n[i]
syn.append(2*np.random.random((nant,1)) - 1)
                        
for i in range(60000):
    #Forward Propagation
    l=[]
    l.append(sigmoid(np.dot(X,syn[0])))
    for i in range (1,nlayers):
        l.append(sigmoid(np.dot(l[i-1],syn[i])))
    
    #Backpropagation
    #Error = (Real-Estimated)*diff(sigmoid)
    l_delta=[0]*nlayers
    l_delta[-1]=((y-l[-1])*(l[-1]*(1-l[-1])))
    for i in range(nlayers-2,-1,-1):
        l_delta[i]=(l_delta[i+1].dot(syn[i+1].T)*(l[i]*(1-l[i])))
    
    
    #new weights distribution
    for i in range(nlayers-1,0,-1):
        syn[i] += l[i-1].T.dot(l_delta[i])
    syn[0]+=X.T.dot(l_delta[0])

#
print("La estimación es: ")
print('\n'.join(' '.join(str(cell) for cell in row) for row in l[-1]))