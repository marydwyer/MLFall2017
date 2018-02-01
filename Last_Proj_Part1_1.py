import numpy as np
from scipy.stats import norm
import random
import matplotlib.pyplot as plt



def datasamples(mu,var,N): #Generate samples of data
    data0 = np.random.normal(mu[0],var[0],N[0])
    data1 = np.random.normal(mu[1],var[1],N[1])
    data2 = np.random.normal(mu[2],var[2],N[2])
    datax = np.r_[data0,data1,data2]
    return datax

def datapdf(axis,pi,mu,var): #Generate pdf of data
    pdf0 = pi[0]*norm.pdf(axis,mu[0],var[0])
    pdf1 = pi[1]*norm.pdf(axis,mu[1],var[1])
    pdf2 = pi[2]*norm.pdf(axis,mu[2],var[2])
    pdfx = pdf0 + pdf1 +pdf2
    return pdfx
    
def rejection(axis,pdf,k,mup,varp,pi,mud,vard): #Perform rejection sampling
    samples = np.zeros(600)
    samplecount = 0
    while samplecount < 600:
        z0 = random.random()
        kprop = k*norm.pdf(z0,mup,varp)
        u = np.random.uniform(0,kprop)
        ptild = datapdf(z0,pi,mud,vard)
        ptild = float(ptild)
        if u < ptild:
            samples[samplecount] = z0
            samplecount = samplecount + 1
    return samples


# =============================================================================
#
# =============================================================================

#Define axis
axis = np.linspace(0,1,101)

#Create mixture model
mudata, vardata = np.array([[0.15], [0.5], [0.7]]), np.array([[0.07], [0.1], [0.03]])
pidata = [0.2, 0.5, 0.3]
Ndata = np.array((200,200,200))
data, pdf = datasamples(mudata,vardata,Ndata), datapdf(axis,pidata,mudata,vardata)

#Creat proposal distribution
muprop, varprop = 0.5, 0.2
k = 4

#Generate rejection sampling
rejectionsamples = rejection(axis,pdf,k,muprop,varprop,pidata,mudata,vardata)

plt.figure
n, bins, patches = plt.hist(data, len(axis), normed=1)
n1, bins1, patches1 = plt.hist(rejectionsamples,len(axis),normed=1,facecolor='green')
plt.show()
