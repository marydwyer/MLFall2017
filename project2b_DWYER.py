#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:01:38 2017

@author: marydwyer
"""

# Mary Dwyer
# Machine Learning Project 2B
# October 11, 2017 
# Predictive Distribution

from numpy import arange, sin, pi
import math
import numpy as np 
import matplotlib.pyplot as plt
from numpy.linalg import inv


# Generate data
x = np.arange(0,1,.01)
t = arange(0.0, 1.0, 0.01)
xdata, noise = np.random.uniform(0,1,100), np.random.normal(0, .2, 100)
points, points4, points2, points1 = np.zeros(25), np.zeros(4), np.zeros(2), np.zeros(1)


f = np.zeros(25)
for i in range (25):
        f[i] = sin(2*pi*xdata[i]) 
        points[i] = f[i] + noise[i]
for i in range (4):
        f[i] = sin(2*pi*xdata[i]) 
        points4[i] = f[i] + noise[i]
for i in range (2):
        f[i] = sin(2*pi*xdata[i]) 
        points2[i] = f[i] + noise[i]
for i in range (1):
        f[i] = sin(2*pi*xdata[i]) 
        points1[i] = f[i] + noise[i]

# Predictive Distribution
s = 0.1
mean_j = np.arange(0,1,1/9)
alpha = 2
Beta = 25
basis, basis4, basis2, basis1, smallbasis = np.zeros(25), np.zeros(4), np.zeros(2), np.zeros(1), np.zeros(9)

mugraph, mugraph4, mugraph2, mugraph1 = np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100)
vargraph, vargraph4, vargraph2, vargraph1 = np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100)
sigma, sigma4, sigma2, sigma1 = np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100)

# Equation 3.4 
### N = 25 ###
phi_j = np.zeros(0) #Design matrix for 25 data points
for j in range (9):
    for i in range (25):
        numerator = (xdata[i]-mean_j[j])**2
        basis[i] = math.exp(-numerator/(2*s ** 2))
        phi_j = np.append(phi_j,(basis[i]))
phi_j = np.reshape(phi_j,(9,25)) #(9 basis functions by 25 data points)
# Small phi
for j in range (100): #run through each x coordiniate on x axis
    for i in range (9):  #run through 9 basis functions
        num = (x[j]-mean_j[i])**2
        smallbasis[i] = math.exp(-num/(2*s ** 2))
# Predictive Distribution
    Sninv = alpha * np.identity(9) + Beta *(np.dot(phi_j,np.transpose(phi_j)))
    Mn = Beta* np.dot(np.dot((inv(Sninv)), phi_j), points)  
    mugraph[j] = np.dot(np.transpose(Mn),smallbasis) 
#Variance
    vargraph[j] = (1/Beta) + np.dot(np.dot(np.transpose(smallbasis),inv(Sninv)),smallbasis)
    sigma[j] = (vargraph[j])**.5
    
### N = 4 ###
phi_j = np.zeros(0) #Design matrix for 4 data points
for j in range (9):
    for i in range (4):
        numerator4 = (xdata[i]-mean_j[j])**2
        basis4[i] = math.exp(-numerator4/(2*s ** 2))
        phi_j = np.append(phi_j,(basis4[i]))
phi_j = np.reshape(phi_j,(9,4)) #(9 basis functions by 4 data points)

# Small phi
for j in range (100): #run through each x coordiniate on x axis
    for i in range (9):  #run through 9 basis functions
        num = (x[j]-mean_j[i])**2
        smallbasis[i] = math.exp(-num/(2*s ** 2))
# Predictive Distribution
    Sninv4 = alpha * np.identity(9) + Beta *(np.dot(phi_j,np.transpose(phi_j)))
    Mn4 = Beta* np.dot(np.dot((inv(Sninv4)), phi_j), points4)  
    mugraph4[j] = np.dot(np.transpose(Mn4),smallbasis) 
#Variance
    vargraph4[j] = (1/Beta) + np.dot(np.dot(np.transpose(smallbasis),inv(Sninv4)),smallbasis)
    sigma4[j] = (vargraph4[j])**.5
    
### N = 2 ###
phi_j = np.zeros(0)
for j in range (9):
    for i in range (2):
        numerator2 = (xdata[i]-mean_j[j])**2
        basis2[i] = math.exp(-numerator2/(2*s ** 2))
        phi_j = np.append(phi_j,(basis2[i]))
phi_j = np.reshape(phi_j,(9,2))
# Small phi
for j in range (100): #run through each x coordiniate on x axis
    for i in range (9):  #run through 9 basis functions
        num = (x[j]-mean_j[i])**2
        smallbasis[i] = math.exp(-num/(2*s ** 2))
# Predictive Distribution
    Sninv2 = alpha * np.identity(9) + Beta *(np.dot(phi_j,np.transpose(phi_j)))
    Mn2 = Beta* np.dot(np.dot((inv(Sninv2)), phi_j), points2)  
    mugraph2[j] = np.dot(np.transpose(Mn2),smallbasis) 
#Variance
    vargraph2[j] = (1/Beta) + np.dot(np.dot(np.transpose(smallbasis),inv(Sninv2)),smallbasis)
    sigma2[j] = (vargraph2[j])**.5  
    
### N = 1 ###
phi_j = np.zeros(0)
for j in range (9):
    for i in range (1):
        numerator1 = (xdata[i]-mean_j[j])**2
        basis1[i] = math.exp(-numerator1/(2*s ** 2))
        phi_j = np.append(phi_j,(basis1[i]))
phi_j = np.reshape(phi_j,(9,1))
# Small phi
for j in range (100): #run through each x coordiniate on x axis
    for i in range (9):  #run through 9 basis functions
        num = (x[j]-mean_j[i])**2
        smallbasis[i] = math.exp(-num/(2*s ** 2))
# Predictive Distribution
    Sninv1 = alpha * np.identity(9) + Beta *(np.dot(phi_j,np.transpose(phi_j)))
    Mn1 = Beta* np.dot(np.dot((inv(Sninv1)), phi_j), points1)  
    mugraph1[j] = np.dot(np.transpose(Mn1),smallbasis) 
#Variance
    vargraph1[j] = (1/Beta) + np.dot(np.dot(np.transpose(smallbasis),inv(Sninv1)),smallbasis)
    sigma1[j] = (vargraph1[j])**.5  


#### Graphs ####
plt.figure(1)

#Graph 1
plt.subplot(2,2,1) #rows,col,item
plt.plot(t, sin(2*pi*t),'g')
plt.ylim((-1.5, 1.5))
plt.ylabel('x')
plt.ylabel('t')
plt.title('N=1')
plt.plot(x,mugraph1,'r--')
plt.plot(xdata[0],points[0],'bo') 
plt.fill_between(x, mugraph1 - sigma1, mugraph1 + sigma1,facecolor='pink')

#Graph 2
plt.subplot(2,2,2)
plt.plot(t, sin(2*pi*t),'g')
plt.ylim((-1.5, 1.5))
plt.ylabel('x')
plt.ylabel('t')
plt.title('N=2')
plt.plot(x,mugraph2,'r--')
for i in range (2):
    plt.plot(xdata[i],points[i],'bo') 
plt.fill_between(x, mugraph2 - sigma2, mugraph2 + sigma2,facecolor='pink')

#Graph 3
plt.subplot(2,2,3)
plt.plot(t, sin(2*pi*t),'g')
plt.ylim((-1.5,1.5))
plt.ylabel('x')
plt.ylabel('t')
plt.title('N=4')
plt.plot(x,mugraph4,'r--')
for i in range (4):
    plt.plot(xdata[i],points[i],'bo') 
plt.fill_between(x, mugraph4 - sigma4, mugraph4 + sigma4,facecolor='pink')

# Graph 4
plt.subplot(2,2,4)
plt.plot(t, sin(2*pi*t),'g')
plt.grid(True)
plt.ylim((-1.5, 1.5))
plt.ylabel('x')
plt.ylabel('t')
plt.title('N=25')
for i in range (25):
    plt.plot(xdata[i],points[i],'bo') 
plt.plot(x,mugraph,'r--')
plt.fill_between(x, mugraph - sigma, mugraph + sigma,facecolor='pink')
    
plt.tight_layout()

