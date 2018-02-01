# Mary Dwyer
# Machine Learning Project 1
# September 27, 2017 
# Implement Conjugate Estimators: binomial, gaussian with known variance, gaussian with known mean

import matplotlib.pyplot as plt
import numpy as np
import math
 
# Generate random data from Normal (Gaussian Distribution)
mu_true = 0
sigmasq_true = 2

a = 10
N = 100
mu = np.zeros(N)
sigmasq = np.zeros(N)
mse = np.zeros(N)
inside_sum = np.zeros(N)
mse_matrix=[]


for j in range(a):
    x = np.random.normal(mu_true, sigmasq_true, N)
    mse = np.zeros(N)

#Equation 1.55 & 1.56: Max Gaussian log likelihood
    for i in range (N):
        mu[i]= (1/(i+1))*sum(x[0:i])
        mse[i] = (mu_true - mu[i])**2
        
    for i in range (N):
        inside_sum[i] = (x[i] - mu[i])**2
        sigmasq[i]= (1/(i+1))*(sum(inside_sum[0:i]))   
    mse_matrix += [mse]


ave_list = [] # Create list of average mse

for j in range(N):
    ave_nums = [] # Average of mse arrays for a given # of data points
    for i in range(a):
        ave_nums.append(mse_matrix[i][j]) # Add the Nth indicies from each mse array
    ave = sum(ave_nums)/len(ave_nums) # Average the indicies from each array
    ave_list.append(ave) # Add average for a given N to the master ave_list


    
##2 CONJUGATE PRIOR BINOMIAL
# Binomial data, binomial liklihood, beta prior to obtain a beta posterior.

# Generate random data
mse2 = np.zeros(N)
mse_matrix2=[]
for j in range(a):
    n, p_true = 10, .4 # (itterations per trial, truth)
    N = 100 # number of tirals
    x2 = np.random.binomial(n, p_true, N) # binomial distribution is fully defined by n, p_true, N
# Guess hyperparameters (BS)
    alpha_guess = .2
    beta_guess = .5
# Result of flipping a coin n times, tested N times
# Gives number of sucesses
    alpha = np.zeros(N)
    betap = np.zeros(N)
    p = np.zeros(N)

    alpha[0]= alpha_guess + x2[0]
    betap[0] = beta_guess + n - x2[0]
    mse2 = np.zeros(N)
    for i in range (1 , N):
        alpha[i] = alpha[i-1] + sum(x2[0:i])
        betap[i] = betap[i-1] + 10*(i+1) - sum(x2[0:i])
        
    
 #Posterior      
    p_out = np.zeros(N)
    for i in range (0 , N):
         p_out[i] = alpha[i] / (alpha[i] + betap[i])
         mse2[i] = (p_true - p_out[i])**2
         
    mse_matrix2 += [mse2]

ave_list2 = [] # Create list of average mse

for j in range(N):
    ave_nums2 = [] # Average of mse arrays for a given # of datapoints
    for i in range(a):
        ave_nums2.append(mse_matrix2[i][j])   # Add the Nth indicies from each mse array
    ave = sum(ave_nums2)/len(ave_nums2) # Average the indicies from each array
    ave_list2.append(ave)    # Add average for a given N to the master ave_list

##3 Conjugate prior of Gaussian with known variance (estimate the mean)
# Generate random data
mse_matrix3=[]
for j in range(a):
    mu_true3 = 2
    sigmasq_known3 = .4

    x3 = np.random.normal(mu_true3, sigmasq_known3, N)

    mu_naught = 1
    sigmasq_naught = .1


    mu3 = np.zeros(N)
    sigmasq3 = np.zeros(N)
    mse3 = np.zeros(N)
    
    mu3[0] = ((mu_naught/sigmasq_naught) + (x3[0]/sigmasq_known3))/((1/sigmasq_naught)+(1/sigmasq_known3))
    sigmasq3[0] = ((1/sigmasq_naught)+(1/sigmasq_known3))**(-1)
    for i in range (1 , N):
        mu3[i] = ((mu_naught/sigmasq_naught) + ((sum(x3[0:(i+1)]))/sigmasq_known3))/((1/sigmasq_naught)+(i/sigmasq_known3))
        sigmasq3[i] = ((1/sigmasq_naught)+(i/sigmasq_known3))**(-1)
        mse3[i] = (mu_true3 - mu3[i])**2

    mse_matrix3 += [mse]


ave_list3 = []   #Create list of average mse
for j in range(N):
    ave_nums3 = []       #Average of mse arrays for a given # of datapoints
    for i in range(a):
        ave_nums3.append(mse_matrix3[i][j])   #Add the Nth indicies from each mse array
    ave3 = sum(ave_nums3)/len(ave_nums3) #Average the indicies from each array
    ave_list3.append(ave)    #Add average for a given N to the master ave_list


##4 Conjugate prior Gaussian with known mean (estimate the variance)
mse_matrix4=[]
for j in range(a):
    mu_known4 = 2
    mse4 = np.zeros(N)

    sigmasq_true4 = .1

    x4 = np.random.normal(mu_known4, sigmasq_true4, N)


# Guess hyperparameters
    alpha_guess4 = 2
    beta_guess4 = 3
    alpha4 = np.zeros(N)
    betap4 = np.zeros(N)
    inside4 = np.zeros(N)

    alpha4[0]= alpha_guess4 + 1/2
    betap4[0] = beta_guess4 + ((x4[0]-mu_known4)**2)/2

    for i in range (1 , N):
        alpha4[i]= alpha4[i-1] + i/2
        inside4[i] = (x4[i-1] - mu_known4)**2
        betap4[i] = betap4[i-1] + (sum(inside4[0:i]))/2
    
    mu_out = np.zeros(N)
    for i in range (0 , N):
        mu_out[i] = ((betap4[i]) **2)/ alpha4[i]
        mse4[i] = (sigmasq_true4 - mu_out[i])**2
    mse_matrix4 += [mse]
ave_list4 = []   # Create list of average mse

##Graphs
#1 Mean Squared Error'
for j in range(N):
    ave_nums4 = []       # Average of mse arrays for a given # of datapoints
    for i in range(a):
        ave_nums4.append(mse_matrix[i][j])   # Add the Nth indicies from each mse array
    ave4 = sum(ave_nums4)/len(ave_nums4) # Average the indicies from each array
    ave_list4.append(ave4)    # Add average for a given N to the master ave_list

datapoints = np.arange(1,N+1,1)
ax = plt.subplot(111)

plt.plot(datapoints,ave_list,datapoints,ave_list2,datapoints,ave_list3,datapoints,ave_list4)
plt.legend(('Model length', 'Data length', 'Total message length'),
           'upper center', shadow=True, fancybox=False)
plt.xlabel("Number of Datapoints")
plt.ylabel("Mean Square Error")
plt.show()

#2 Binomial Conjugate
from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
xgraph2 = np.arange (0.01, 1, 0.01)

a =  alpha[9]
b = betap[9]
y = beta.pdf(xgraph2,a,b)
p2 = plt.figure()
plt.plot(xgraph2,y)
p2.suptitle('Binomial with 10 observations')
plt.xlabel('Converges to 0.4')
plt.show

a_2 =  alpha[49]
b_2 = betap[49]
y_2 = beta.pdf(xgraph2,a_2,b_2)
p2_2 = plt.figure()
plt.plot(xgraph2,y_2)
p2_2.suptitle('Binomial with 50 observations')
plt.xlabel('Converges to 0.4')
plt.show

a_3 =  alpha[99]
b_3 = betap[99]
y_3 = beta.pdf(xgraph2,a_3,b_3)
p2_3 = plt.figure()
plt.plot(xgraph2,y_3)
p2_3.suptitle('Binomial with 100 observations')
plt.xlabel('Converges to 0.4')
plt.show

#2 Binomial Conjugate with new hyperparameters
alpha_guess = 10
beta_guess = 20

a_4 =  alpha[99]
b_4 = betap[99]
y_4 = beta.pdf(xgraph2,a_3,b_3)
p2_4 = plt.figure()
plt.plot(xgraph2,y_4)
p2_4.suptitle('Binomial with 100 observations and Large Hyperparamters')
plt.xlabel('Converges to 0.4')
plt.show


#3 Gaussian with Known Variance
import matplotlib.mlab as mlab
xgraph3 = np.arange (0.01, 4, 0.01)

mugraph3 = mu3[9]
variancegraph3 = sigmasq3[9]
sigmagraph3 = math.sqrt(variancegraph3)
p3_1 = plt.figure()
plt.plot(xgraph3,mlab.normpdf(xgraph3, mugraph3, sigmagraph3))
p3_1.suptitle('Gaussian with Known Variance, x = 10')
plt.xlabel('Converges to 2.0')
plt.show

mugraph3_2 = mu3[49]
variancegraph3_2 = sigmasq3[49]
sigmagraph3_2 = math.sqrt(variancegraph3_2)
p3_2 = plt.figure()
plt.plot(xgraph3,mlab.normpdf(xgraph3, mugraph3_2, sigmagraph3_2))
p3_2.suptitle('Gaussian with Known Variance, x = 50')
plt.xlabel('Converges to 2.0')
plt.show

mugraph3_3 = mu3[99]
variancegraph3_3 = sigmasq3[99]
sigmagraph3_3 = math.sqrt(variancegraph3_3)
p3_3 = plt.figure()
plt.plot(xgraph3,mlab.normpdf(xgraph3, mugraph3_3, sigmagraph3_3))
p3_3.suptitle('Gaussian with Known Variance, x = 100')
plt.xlabel('Converges to 2.0')
plt.show

#4 Gaussian with Known Mean
from scipy.stats import gamma
xgraph4 = np.arange (-0.2,0.2,0.01)

mu4 = mu_out[9]
y4 = gamma.pdf(xgraph4,mu4)
p4 = plt.figure()
plt.plot(xgraph4,y4)
p4.suptitle('Gaussian with Known Mean, x=10')
plt.xlabel('Converges to 0.1')
plt.show

mu4_2 = mu_out[49]
y4_2 = gamma.pdf(xgraph4,mu4_2)
p4_2 = plt.figure()
plt.plot(xgraph4,y4_2)
p4_2.suptitle('Gaussian with Known Mean, x=50')
plt.xlabel('Converges to 0.1')
plt.show

mu4_2 = mu_out[99]
y4_2 = gamma.pdf(xgraph4,mu4_2)
p4_2 = plt.figure()
plt.plot(xgraph4,y4_2)
p4_2.suptitle('Gaussian with Known Mean, x=50')
plt.xlabel('Converges to 0.1')
plt.show