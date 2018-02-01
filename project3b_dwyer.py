#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:38:56 2017

@author: marydwyer
"""

#Logistic Regression Classifier using IRLS Algorithm
import numpy 
import numpy as np

#===============================================
# Creating data sets

# Given
Mu1 = np.array([1,1])      
Mu2 = np.array([-1,-1])
sigma = np.identity(2)
#Number of Data points
#==============================================================================
N = 200
N1 = numpy.random.binomial(N, 0.5)
N2 = N - N1
#==============================================================================


# Multivariates
data1 = numpy.random.multivariate_normal(Mu1,sigma,N1) #(meanvector, covariance, number of data sets)
data2 = numpy.random.multivariate_normal(Mu2,sigma,N2) 

testdata1 = numpy.random.multivariate_normal(Mu1,sigma,N1)
testdata2 = numpy.random.multivariate_normal(Mu2,sigma,N2)

# Vertically stack the two data sets
data = np.vstack((data1,data2))
testdata = np.vstack((testdata1,testdata2))

# t values
t1 = np.ones(100)
t2 = np.zeros(100)
t = np.vstack((t1,t2))
t = np.reshape(t,(200,1))

# Design Matrix
one200 =np.ones(200)
design  = np.c_[one200,data]

# Guess initial weights
w = np.array([[.3],[.5],[.7]])

# Initialize
designt = np.zeros((0))
t0 = ()

# Train the data (IRLS Algorithm)
for i in range (0,200):

    designt = numpy.append(designt,design[i,:]) #each iteration add a row from the design matrix
    designt= designt[:,None] #I needed this line to make the matrix the shape i wanted
    newdesign = np.reshape(designt, (i+1, 3))
    phi = np.transpose(newdesign)

    a = np.dot(np.transpose(w), phi) #this gets plugged into sigmoid
    y = 1/(1+np.exp(-a))
    y0 = np.transpose(y) 
    R = numpy.diagflat(y) 
  
    t0 = numpy.append(t0,t[i]) 
    t0 = t0[:,None]
    t_use = np.transpose(t0)
    diff= y0-t0
    
    Z = (newdesign @ w) - (np.linalg.pinv(R) @ (diff))
 
    w = np.linalg.pinv(phi @ R @ newdesign) @ phi @ R @ Z #new weights
    
   
# Test classifier with newest weights
design_test  = np.c_[one200,testdata] # make a design matrix for test data
a = np.transpose(w) @ np.transpose(design_test)

#Check percentage correct
Pc1 = 1/(1+np.exp(-a))
pc1 = np.transpose(Pc1)
k=0
for j in range (N1):
    if pc1[j] >.5:
        k=k+1
for j in range ((N2-N1),200):
    if pc1[j] <.5:
        k=k+1

correct = (k/200)*100
print (correct)


    
