 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 11:47:26 2017

@author: marydwyer
"""


# Mary Dwyer
# Machine Learning Project 5
# November 12, 2017 
# Expectation Maximization

# Implement EM on a Gaussian mixture model in 1 and 2 dimensions with K= 3. 
# The choice of means, covariance and pi is up to you.  
# The algorithm is laid out explicitly in equations 9.23-9.28.

# For the 1-d case, produce a plot that shows a histogram of your generated 
# observations, and overlay on that histogram the pdf you found. 
# Plot this at algorithm init, and a couple other times as the algorithm converges

# For 2-D, create a plot similar to 9.8, but with K = 3. 
import math
import numpy as np 
import matplotlib.pyplot as plt
#from numpy.linalg import inv
from scipy.stats import multivariate_normal
from scipy.stats import norm

def computegamma(k,x,pik,mu,sig):
    num = pik[k] * multivariate_normal.pdf(x,mu[k],sig[k])
    denom = 0
    for i in range (0,3):
        denom = denom + pik[i] * multivariate_normal.pdf(x,mu[i],sig[i])
    return num/denom

def updateNk(k,gamma):
    return np.sum(gamma[:,k])

def updateMuk(k,x,N,Nk,gamma):
    sum = 0
    for i in range (0,N):
        sum = sum + gamma[i,k] * x[i]
    return 1/Nk[k] * sum

def updateSigma(k,x,N,Nk,muk,gamma,dim):
    sum = np.zeros((dim,dim))
    for i in range (0,N):
        first = x[i] - muk[k] 
        sec = np.transpose(first.reshape(1,dim))
        sum = sum + gamma[i,k]*np.dot(first.reshape(1,dim).T,sec.T)
    return 1/Nk[k] * sum

def tri_norm(x,m1, m2, m3, s1, s2, s3, k1, k2, k3):
    ret = k1*norm.pdf(x, loc=m1 ,scale=s1)
    ret += k2*norm.pdf(x, loc=m2 ,scale=s2)
    ret += k3*norm.pdf(x, loc=m3 ,scale=s3)
    return ret

# 1d Case 
# Generate Data
K =3
Mu1 = np.array([-3])      
Mu2 = np.array([3])
Mu3 = np.array([0])
sigma = [0.5*np.identity(1)]*K

N = 1000
P = [0.2, 0.3, 0.5]
multi = np.random.multinomial(N,P)

data1 = np.random.multivariate_normal(Mu1,sigma[0],multi[0]) #(meanvector, covariance, number of data sets)
data2 = np.random.multivariate_normal(Mu2,sigma[1],multi[1]) 
data3 = np.random.multivariate_normal(Mu3,sigma[2],multi[2])
datacon = np.concatenate((data1,data2),axis=0)
data = np.concatenate((datacon,data3),axis=0)

#Generate random pik
pik = np.random.multinomial(N,[1/K]*K)/N

#Iteration
gamma = np.zeros((N,K))
Nk = np.zeros((K,1))
Mu = np.zeros((K,1))
j = 0
for i in range (0,K):
    Mu[i] = np.random.uniform(0,1,1)
for i in range (0,25):
    print(i)
    for n in range (0,N):
        for k in range (0,K):
            gamma[n,k]=computegamma(k,data[n],pik,Mu,sigma)

    for k in range (0,K):
        Nk[k] = updateNk(k,gamma)
        Mu[k] = updateMuk(k,data,N,Nk,gamma)
        sigma[k] = updateSigma(k,data,N,Nk,Mu,gamma,1)
        pik[k] = Nk[k]/N
    if i+1 in [1,2,3,5,10,25]:
        j=j+1
        plt.subplot(3,2,j)
        plt.hist(data)
        muu = float(Mu[0])
        sigmaa = float(sigma[0])
        xx = np.linspace(muu - 3*sigmaa, muu + 3*sigmaa, 100)
        plt.plot(xx,450*norm.pdf(xx, muu, sigmaa))
        plt.xlim(-5,5)
        
        muu = float(Mu[1])
        sigmaa = float(sigma[1])
        xx = np.linspace(muu - 3*sigmaa, muu + 3*sigmaa, 100)
        plt.plot(xx,450*norm.pdf(xx, muu, sigmaa))
        plt.xlim(-5,5)
        
        muu = float(Mu[2])
        sigmaa = float(sigma[2])
        xx = np.linspace(muu - 3*sigmaa, muu + 3*sigmaa, 100)
        plt.plot(xx,450*norm.pdf(xx, muu, sigmaa))
        plt.xlim(-5,5)
    if i+1 == 25:
        plt.show()
        


# 2d Case
# Generate Data
K =3
Mu1 = np.array([-3,0])      
Mu2 = np.array([3,0])
Mu3 = np.array([0,3])
sigma = [0.5*np.identity(2)]*K

N = 1000
P = [0.2, 0.3, 0.5]
multi = np.random.multinomial(N,P)

data1 = np.random.multivariate_normal(Mu1,sigma[0],multi[0]) #(meanvector, covariance, number of data sets)
data2 = np.random.multivariate_normal(Mu2,sigma[1],multi[1]) 
data3 = np.random.multivariate_normal(Mu3,sigma[2],multi[2])
datacon = np.concatenate((data1,data2),axis=0)
data = np.concatenate((datacon,data3),axis=0)

#Generate random pik
pik = np.random.multinomial(N,[1/K]*K)/N

#Iteration
gamma = np.zeros((N,K))
Nk = np.zeros((K,1))
Mu = np.zeros((K,2))
for i in range (0,K):
    Mu[i,0] = np.random.uniform(0,1,1)
    Mu[i,1] = np.random.uniform(0,1,1)
j = 0

for i in range (0,25):
    print(i)
    for n in range (0,N):
        for k in range (0,K):
            gamma[n,k]=computegamma(k,data[n],pik,Mu,sigma)

    for k in range (0,K):
        Nk[k] = updateNk(k,gamma)
        Mu[k] = updateMuk(k,data,N,Nk,gamma)
        sigma[k] = updateSigma(k,data,N,Nk,Mu,gamma,2)
        pik[k] = Nk[k]/N

    if i+1 in [1,2,3,5,10,25]:
        j=j+1
        plt.subplot(3,2,j)
        for l in range (0,N):
            plt.scatter(data[l,0],data[l,1],c=gamma[l])
    if i+1 == 25:
        plt.show()
        
 
        
    
        

