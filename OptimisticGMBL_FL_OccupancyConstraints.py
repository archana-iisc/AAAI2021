#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 14:01:21 2020

@author: archu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:24:52 2020

@author: archu
"""


#Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from UtilityMethods import utils
import sys
import gym
import os

import time

start_time = time.time()


HPC = False

#Initialize:
EPISODE_LENGTH = 10
NUMBER_SIMULATIONS = 10
EPS = 0.01
DELTA = 0.05
N_STATES = 9
N_ACTIONS = 4
RUN_NUMBER = 1000


R = np.zeros((EPISODE_LENGTH, N_STATES, N_ACTIONS))
C = np.zeros((EPISODE_LENGTH,N_STATES, N_ACTIONS))
P = np.zeros((N_STATES,N_ACTIONS,N_STATES))

if HPC == False:
    RUN_NUMBER = 1729
else:
    RUN_NUMBER = int(os.environ.get('BASHVAR'))
    
P[0, 0, 0] = 1
P[0, 1, 0] = 1/3
P[0, 1, 1] = 2/3
P[0, 2, 0] = 1/3
P[0, 2, 3] = 2/3
P[0, 3, 0] = 1

P[1, 0, 0] = 2/3
P[1, 0, 1] = 1/3
P[1, 1, 1] = 1/3
P[1, 1, 2] = 2/3
P[1, 2, 1] = 1/3
P[1, 2, 4] = 2/3
P[1, 3, 1] = 1

P[2, 0, 2] = 1/3
P[2, 0, 1] = 2/3
P[2, 1, 2] = 1
P[2, 2, 2] = 1/3
P[2, 2, 5] = 2/3
P[2, 3, 2] = 1

P[3, 0, 3] = 1
P[3, 1, 3] = 1/3
P[3, 1, 4] = 2/3
P[3, 2, 3] = 1/3
P[3, 2, 6] = 2/3
P[3, 3, 3] = 1/3
P[3, 3, 0] = 2/3


### Lets say state 4 is hole:
P[4, 0, 4] = 0.1
P[4, 0, 3] = 0.9
P[4, 1, 4] = 0.1
P[4, 1, 5] = 0.9
P[4, 2, 4] = 0.1
P[4, 2, 7] = 0.9
P[4, 3, 4] = 0.1
P[4, 3, 1] = 0.9


P[5, 0, 4] = 2/3
P[5, 0, 5] = 1/3
P[5, 1, 5] = 1
P[5, 2, 5] = 1/3
P[5, 2, 8] = 2/3
P[5, 3, 2] = 2/3
P[5, 3, 5] = 1/3

P[6, 0, 6] = 1
P[6, 1, 6] = 1/3
P[6, 1, 7] = 2/3
P[6, 2, 6] = 1
P[6, 3, 3] = 2/3
P[6, 3, 6] = 1/3

P[7, 0, 6] = 2/3
P[7, 0, 7] = 1/3
P[7, 1, 7] = 1/3
P[7, 1, 8] = 2/3
P[7, 2, 7] = 1
P[7, 3, 4] = 2/3
P[7, 3, 7] = 1/3

## State 8 is the GOAL!!
P[8, 0, 8] = 1
P[8, 1, 8] = 1
P[8, 2, 8] = 1
P[8, 3, 8] = 1
print(np.sum(P,axis=2))
print(P[0,0,0])

#Costs
C[:,1,2] = 1
C[:,3,1] = 1
C[:,5,0] = 1
C[:,7,3] = 1
C[:,4,:] = 1

#Rewards
R[:,5,2] = 1
R[:,7,1] = 1
R[:,8,:] = 1

CONSTRAINT = 1.0

LIST_NUMBER_SAMPLES = np.arange(1,200,step=1)
LIST_STEPS = [item*36 for item in LIST_NUMBER_SAMPLES]

# NUMBER_OF_SAMPLES = int((1/(EPS**2))*np.log(2/DELTA**2))



STATES = np.arange(N_STATES)
ACTIONS = np.arange(N_ACTIONS)

util_methods = utils(EPS, DELTA, 0, P,R,C,EPISODE_LENGTH,N_STATES,N_ACTIONS,CONSTRAINT)
opt_policy_1, opt_value_LP, opt_cost_LP = util_methods.compute_opt_LP_Constrained()



VALUE_GAP = np.zeros(len(LIST_NUMBER_SAMPLES))
COST_GAP = np.zeros(len(LIST_NUMBER_SAMPLES))
Value = np.zeros((len(LIST_NUMBER_SAMPLES),NUMBER_SIMULATIONS))
Cost = np.zeros((len(LIST_NUMBER_SAMPLES),NUMBER_SIMULATIONS))

for l in range(NUMBER_SIMULATIONS):
 #np.random.seed(l+1)
 k = 0
 for num_samples in LIST_NUMBER_SAMPLES:
    print("printing n \n")
    print(num_samples)
    k += 1
    util_methods = utils(EPS,DELTA,0,P,R,C,EPISODE_LENGTH,N_STATES,N_ACTIONS,CONSTRAINT)
    #util_methods.setQvals(qVals1,opt_policy)
    
    NUMBER_OF_OCCURANCES_P = np.zeros((N_STATES,N_ACTIONS,N_STATES))
    
    P_hat = np.zeros((N_STATES,N_ACTIONS,N_STATES))
    BETA = np.zeros((N_STATES,N_ACTIONS,N_STATES))
    
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            for n in range(num_samples):
                probs = list(P[s,a,:])
                next_s = int(np.random.choice(STATES,1,replace=True,p=probs))
                NUMBER_OF_OCCURANCES_P[s,a,next_s] += 1
    print("printing n \n")
    print(num_samples)          
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            for next_s in range(N_STATES):
                P_hat[s,a,next_s] = NUMBER_OF_OCCURANCES_P[s,a,next_s]/(max(num_samples,1))
                BETA[s,a,next_s] = min(np.sqrt(P_hat[s,a,next_s]*(1-P_hat[s,a,next_s])/num_samples) + 1/(max(num_samples,1)), 1/np.sqrt(max(num_samples,1)))
                
    util_methods.setEmpiricalKernel(P_hat)
    
    util_methods.setConfidenceIntervals(BETA)
                
    pi_k, value, cost_of_policy = util_methods.compute_extended_LP(1)
    
    Value[k-1,l] = abs(value[0,0] - opt_value_LP[0,0])
    Cost[k-1,l] = abs(cost_of_policy[0,0] - CONSTRAINT)
   
val = np.mean(Value,axis=1)
cost = np.mean(Cost,axis=1)
val_std = np.std(Value,axis=1)
cost_std = np.std(Cost,axis=1)
        
      
# values_gap_mean = np.mean(VALUE_GAP,axis=0)
df = pd.DataFrame({'value_gap_mean':val,'constraint_gap_mean':cost,'value_gap_std':val_std,'cost_gap_std':cost_std})
df.to_csv("Value_ConstrainedLP_GMBL_confidence_10.csv",header=None)

if HPC == False:
    plt.figure(1)
    plt.plot(LIST_STEPS,val,'r',label="Objective")
    plt.xlabel("Number of Samples")
    plt.ylabel("Value difference")
    plt.legend(loc="upper left")
    plt.title('Value Convergence for GMBL, LP')
    #plt.savefig('GMBL convergence,FL Value.pdf')
    plt.show()

    plt.figure(2)
    plt.plot(LIST_STEPS,cost,'b',label="Objective")
    plt.legend(loc="upper left")
    plt.xlabel("Number of Samples")
    plt.ylabel("Constraint Value difference")
    plt.title('Constraint Convergence for GMBL, LP')
    #plt.savefig('GMBL convergence,FL Constraint.pdf')
    plt.show()

print("--- %s seconds ---" % (time.time() - start_time))

