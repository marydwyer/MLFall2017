
#Part 2 
#MCMC

import numpy 
import numpy  as np
import matplotlib.pyplot as plt

#generate uniform x values
x = numpy.random.uniform(-1,1,(250,1))

#generate noise
noise_sig = 0.2
noise_mean = 0.0
noise = numpy.random.normal(noise_mean,noise_sig,(250,1))

#Observations
w0_true = .2 #These are what we're looking to predict
w1_true = .5
w_sigma = 1.0 #prior SD of weights
t = x* w1_true + w0_true +noise

#Start Monte Carlo
#Define liklihood and proir (Our choise)
#Gaussian prior,Gaussian Likilihood
#Posterior will be Gaussian,but we dont care about the normalization constant

def prior(w,w_sigma): #p(w) probabliity of weight
    p = (1/np.sqrt(2*np.pi**2)*w_sigma**2)*np.exp(-(w[0]**2 +w[1]**2)/(2*w_sigma**2))
    return p 

def liklihood(t,x,w,noise_sig,noise_mean): #liklihood is prob t given w, x 
    like = (1/np.sqrt(2*np.pi)*noise_sig)*np.exp(-(w[0] +w[1]*x - t -noise_mean)**2/(2*noise_sig**2))
    return like

def logposterior(t,x,w,w_sigma,noise_sig,noise_mean): #log posterior so we dont have rounding errors
    log_prior = np.log(prior(w,w_sigma))
    log_like= np.sum(np.log(liklihood(t,x,w,noise_sig,noise_mean)))
    log_posterior = log_prior+log_like #since we log, the product turns into a sum
    return log_posterior

#Monte Carlo 
N = 1000000

#p(z_tau) = calculate this previouis
#compute A of these, need to log this also
w = np.array([0,0]) #This is where we start, we will burn because we want .2,.5
accepted_samples = [] #Add to this later
for i in range (N):
    #sample from proposed  noraml, generate two independent values
    if i % 1000 == 0:
        print(i/N) #gives a count
    proposal = np.random.normal(size=(1,2),scale = 1)[0]  #scale means SD

    pz_star= logposterior(t,x,proposal,w_sigma,noise_sig,noise_mean) #p(z*) = calculate random sample from proposal, DO WE KEEP THIS OR NOT
    pz_tau=logposterior(t,x,w,w_sigma,noise_sig,noise_mean) #previous sample, w is temporary and keeps changing
    
    #A is min of Pz*/Pztau, but we logged the whole thing so A = min (0, logpz*-logpz*)
    A = np.minimum(0.0,pz_star-pz_tau)
    u = np.random.uniform(0,1)
    if A >= np.log(u) :
        w = proposal #the proposal is appended to the new list
    accepted_samples.append(w) #or the previous sample is 

#plots
W = np.array(accepted_samples) #numpy array so we can work with it

w = np.mean(W[1000:,:],0)#Burning throw away first 100 samples # 0 axis  go down vertically this is the answer AND take the mean
print(w)
xx = np.linspace(0,.5,25)
yy = np.linspace(0,.7,25)
X,Y = np.meshgrid(xx,yy)
histogram = np.histogram2d(W[:,0], W[:,1],bins = 25,range = [[0,1],[0,1]])
plt.contourf(X,Y,histogram[0].T)
plt.show()

