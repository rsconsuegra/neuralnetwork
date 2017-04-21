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


n=int(input('Numero de Neuronas de la primera capa='))
n2=int(input('Numero de Neuronas de la segunda capa='))

#Observations and real values: X values to estimate, Y real values.
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T

#weight values
syn0 = 2*np.random.random((3,n)) - 1
syn1 = 2*np.random.random((n,n2)) - 1
syn2 = 2*np.random.random((n2,1)) - 1
                         
for j in range(60000):

    #Forward Propagation
    l1 = sigmoid(np.dot(X,syn0))
    l2 = sigmoid(np.dot(l1,syn1))
    l3 = sigmoid(np.dot(l2,syn2))
    
    #Backpropagation
    #Error = (Real-Estimated)*diff(sigmoid)
    l3_delta = (y - l3)*(l3*(1-l3)) 
    l2_delta = l3_delta.dot(syn2.T) * (l2 * (1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    
    #new weights distribution
    syn2 += l2.T.dot(l3_delta)
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)

print("La estimación es: ")
print('\n'.join(' '.join(str(cell) for cell in row) for row in l3))
