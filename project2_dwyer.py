#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:26:24 2017

@author: marydwyer
"""

import numpy as np 
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv
from scipy.stats import multivariate_normal

## Generate synthetic data from the function f(x,a) = a0 + a1x

# Choose values of x from the uniform distribution U(x|-1,1)
xdata = np.random.uniform(-1,1,200)
# Add Gaussian noise with a standard deviation of 0.2 to obtain the target values tn.
noise = np.random.normal(0, .2**2, 100)
a0, a1 = -0.3, 0.5
f, t = np.zeros(20), np.zeros(20)
for i in range (20):
       f[i] = a0 + a1 * xdata[i]
       t[i] = f[i] + noise[i]

# Data space
w0r, w1r = np.random.normal(0, .5, 100),np.random.normal(0, .5, 100)

## Likelihood = Gaussian (Column 1)
w0l = np.arange(-1, 1, 0.01)
w1l = np.arange(-1, 1, 0.01)
#Generate a normal distribution function for 3 different likelihood inputs 
def normdist(x,mu,sigma):
    var = float(sigma)**2
    norm = math.exp(-(float(x)-float(mu))**2/(2*var))/((2*math.pi*var)**.5)
    return norm
# Likelihood 1: Run through all w values for the 0th index in the target and observation matrices
likely1 = np.zeros((200,200))
for i in range(200):
    for j in range(200):
        likely1[i,j]=normdist(t[0],w0l[i] + w1l[j] * xdata[0], .2)
    
# Likelihood 2: Run through all w values for the 2nd index in the target and observation matrices
likely2 = np.zeros((200,200))
for i in range(200):
    for j in range(200):
        likely2[i,j]=normdist(t[2],w0l[i] + w1l[j] * xdata[2], .2)

# Likelihood 3: Run through all w values for the 10th index in the target and observation matrices
likely3 = np.zeros((200,200)) 
for i in range(200):
    for j in range(200):
        likely3[i,j]=normdist(t[10],w0l[i] + w1l[j] * xdata[10], .2)


## Prior/Posterior
w1 = np.zeros(0)
for i in [float(j) / 100 for j in range(-100, 100, 1)]:
   for k in range (200):
       w1=np.append(w1,i) #Making  abig 200x200 matrix will values so I can make w0 from its transpose
w1 = np.reshape(w1,(200,200))
w0 = np.transpose(w1)
w0flat = w0.flatten() #Flatten into a column
w1flat = w1.flatten()
w=np.c_[w0flat,w1flat] #get them side-by-side

#Multivarite
mean = [0,0]
cov=np.identity(2)
post1 = multivariate_normal.pdf(w,mean,cov)
post1shape = np.reshape(post1,(200,200))

# UPDATES
# GIVEN 
alpha = 2
Beta = 25

# First Update: One Data Point 
basis1 = np.zeros(0)
for i in range (1):
   basis1=np.append(basis1,(1)) #invers of basis function
for i in range (1):
   basis1=np.append(basis1,(xdata[i]))
basis1 = np.reshape(basis1,(2,1))

Sninv1 = alpha * np.identity(2) + Beta *(np.dot(basis1,np.transpose(basis1)))
for i in range (1):
   t1 = t[i]
Mn1 = Beta* np.dot(np.dot((inv(Sninv1)), basis1),t1)
aa = [-.416,.240]
post2 = multivariate_normal.pdf(w,aa,inv(Sninv1))
post2shape = np.reshape(post2,(200,200))

# Second Update: Two Data Points
basis2 = np.zeros(0)
for i in range (2):
   basis2=np.append(basis2,(1))
for i in range (2):
   basis2=np.append(basis2,(xdata[i]))
basis2 = np.reshape(basis2,(2,2))

Sninv2 = alpha * np.identity(2) + Beta *(np.dot(basis2,np.transpose(basis2)))
t2 = np.zeros(2)
for i in range (2):
   t2[i] = t[i]
Mn2 = Beta* np.dot(np.dot((inv(Sninv2)), basis2),t2)
aa2 = [Mn2[0],Mn2[1]]
post3 = multivariate_normal.pdf(w,aa2,inv(Sninv2))
post3shape = np.reshape(post3,(200,200))

# Last Update: 20 Data Points
basis = np.zeros(0)
for i in range (20):
   basis=np.append(basis,(1))
for i in range (20):
   basis=np.append(basis,(xdata[i]))
basis = np.reshape(basis,(2,20))

Sninv = alpha * np.identity(2) + Beta *(np.dot(basis,np.transpose(basis)))
Mn = Beta* np.dot(np.dot((inv(Sninv)), basis),t)

post4 = multivariate_normal.pdf(w,Mn,inv(Sninv))
post4shape = np.reshape(post4,(200,200))

## Graphs
plt.figure(1,figsize=(10,10))

# Right Column: Data Space
plt.subplot(4,3,3) 
x = np.arange(-1,1,.1)
for i in range(6):
   y1 = w0r[i] + w1r[i]* x 
   plt.plot(x,y1,'r')

plt.title('data space')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-1,1)
plt.xlim(-1,1)

# Second Row
plt.subplot(4,3,6) #rows,col,item
new_weightw0_0 = np.random.normal(Mn1[0], .2, 100)
new_weightw1_0 = np.random.normal(Mn1[1], .2, 100)
for i in range(6):
   y1 = new_weightw0_0[i] + new_weightw1_0[i]* x 
   plt.plot(x,y1,'r')

plt.title('data space')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-1,1)
plt.xlim(-1,1)
plt.plot(xdata[0],t[0],'bo') 

# Third Row
plt.subplot(4,3,9) #rows,col,item
new_weightw0_1 = np.random.normal(Mn2[0], .2, 100)
new_weightw1_1 = np.random.normal(Mn2[1], .2, 100)
for i in range(7,13):
   y1 = new_weightw0_1[i] + new_weightw1_1[i]* x 
   plt.plot(x,y1,'r')

plt.title('data space')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-1,1)
plt.xlim(-1,1)
plt.plot(xdata[1],t[1],'bo')
plt.plot(xdata[2],t[2],'bo') 

# Fourth Row
plt.subplot(4,3,12) #rows,col,item
new_weightw0_4 = np.random.normal(Mn[0], .2, 100)
new_weightw1_4 = np.random.normal(Mn[1], .2, 100)
for i in range(7,13):
   y1 = new_weightw0_4[i] + new_weightw1_4[i]* x 
   plt.plot(x,y1,'r')

plt.title('data space')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-1,1)
plt.xlim(-1,1)
for i in range(19):
   plt.plot(xdata[i],t[i],'bo')

# Likelihood
# Second Row
plt.subplot(4,3,4)
L1 = plt.contourf(w0l, w1l, likely1)
plt.xlabel('w0')
plt.ylabel('w1')  
plt.title('Likelihood')
# Third Row
plt.subplot(4,3,7)
L2 = plt.contourf(w0l, w1l, likely2)
plt.xlabel('w0')
plt.ylabel('w1')  
# Fourh Row
plt.subplot(4,3,10)
L3 = plt.contourf(w0l, w1l, likely3)
plt.xlabel('w0')
plt.ylabel('w1') 


# PRIOR/POSTERIOR
plt.subplot(4,3,2)
plt.title('Prior/Posterior')
plt.ylim(-1,1)
plt.xlim(-1,1) 
ax = plt.contourf(w0,w1,post1shape)
plt.xlabel('w0')
plt.ylabel('w1') 

plt.subplot(4,3,5)
plt.ylim(-1,1)
plt.xlim(-1,1)  
ax = plt.contourf(w0,w1,post2shape)
plt.xlabel('w0')
plt.ylabel('w1') 

plt.subplot(4,3,8)
plt.ylim(-1,1)
plt.xlim(-1,1) 
ax = plt.contourf(w0,w1,post3shape)
plt.xlabel('w0')
plt.ylabel('w1') 

plt.subplot(4,3,11)
plt.ylim(-1,1)
plt.xlim(-1,1)
plt.plot(-.3,.5, 'bo-')  
ax = plt.contourf(w0,w1,post4shape)
plt.xlabel('w0')
plt.ylabel('w1') 
plt.tight_layout()