#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 00:07:19 2020

@author: archu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:04:45 2020

@author: archu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 18:23:36 2020

@author: archu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 16:06:44 2020

@author: archu
"""


#Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from UtilityMethods import utils
import gym
import time
import os

start_time = time.time()
HPC = False


#Initialize:
NUMBER_EPISODES = 500
EPISODE_LENGTH = 10
NUMBER_SIMULATIONS = 1
EPS = 0.01
DELTA = 0.05
N_STATES = 9
N_ACTIONS = 4
RUN_NUMBER = 13

if HPC == False:
    RUN_NUMBER = 3
else:
    RUN_NUMBER = int(os.environ.get('BASHVAR'))


R = np.zeros((EPISODE_LENGTH, N_STATES, N_ACTIONS))
C = np.zeros((EPISODE_LENGTH,N_STATES, N_ACTIONS))
P = np.zeros((N_STATES,N_ACTIONS,N_STATES))

#Transitions
P = np.zeros((N_STATES,N_ACTIONS,N_STATES))

# 0 -Left
# 1- Right
# 2 - Down
# 3 - Up

P[0, 0, 0] = 1
P[0, 1, 0] = 1/3 #0
P[0, 1, 1] = 2/3 #1
P[0, 2, 0] = 1/3 #0
P[0, 2, 3] = 2/3 #1
P[0, 3, 0] = 1

P[1, 0, 0] = 2/3 #1
P[1, 0, 1] = 1/3 #0
P[1, 1, 1] = 1/3 #0
P[1, 1, 2] = 2/3 #0
P[1, 2, 1] = 1/3 #0
P[1, 2, 4] = 2/3 #1
P[1, 3, 1] = 1

P[2, 0, 2] = 1/3 #0
P[2, 0, 1] = 2/3 #1
P[2, 1, 2] = 1
P[2, 2, 2] = 1/3 #0
P[2, 2, 5] = 2/3 #1
P[2, 3, 2] = 1

P[3, 0, 3] = 1
P[3, 1, 3] = 1/3 #0
P[3, 1, 4] = 2/3 #1
P[3, 2, 3] = 1/3 #0
P[3, 2, 6] = 2/3 #1
P[3, 3, 3] = 1/3 #0
P[3, 3, 0] = 2/3 #1


### Lets say state 4 is trap state, it will give you a small reward and send you back to where you began:
P[4, :, 0] = 1 


P[5, 0, 4] = 2/3 #1
P[5, 0, 5] = 1/3 #0
P[5, 1, 5] = 1
P[5, 2, 5] = 1/3 #0
P[5, 2, 8] = 2/3 #1
P[5, 3, 2] = 2/3 #1
P[5, 3, 5] = 1/3 #0

P[6, 0, 6] = 1
P[6, 1, 6] = 1/3 #0
P[6, 1, 7] = 2/3 #1
P[6, 2, 6] = 1
P[6, 3, 3] = 2/3 #1
P[6, 3, 6] = 1/3 #0

P[7, 0, 6] = 2/3 #1
P[7, 0, 7] = 1/3 #0
P[7, 1, 7] = 1/3 #0
P[7, 1, 8] = 2/3 #1
P[7, 2, 7] = 1
P[7, 3, 4] = 2/3 #1
P[7, 3, 7] = 1/3 #0

## State 8 is the GOAL!!
P[8, 0, 8] = 1
P[8, 1, 8] = 1
P[8, 2, 8] = 1
P[8, 3, 8] = 1
print(np.sum(P,axis=2))
print(P[0,0,0])

#Costs: "Right" action incurs a cost
C[:,:,1] = 1

#Rewards
R[:,5,2] = 10
R[:,7,1] = 10
R[:,8,:] = 0
R[:,3,1] = 2
R[:,1,2] = 2
R[:,7,3] = 2
R[:,5,0] = 2

CONSTRAINT = 2.0

M = 1024* N_STATES*EPISODE_LENGTH**2/EPS**2
np.random.seed(RUN_NUMBER)

STATES = np.arange(N_STATES)
ACTIONS = np.arange(N_ACTIONS)

ObjRegret = np.zeros((NUMBER_SIMULATIONS,NUMBER_EPISODES))
ConRegret = np.zeros((NUMBER_SIMULATIONS,NUMBER_EPISODES))

util_methods_1 = utils(EPS, DELTA, M, P,R,C,EPISODE_LENGTH,N_STATES,N_ACTIONS,CONSTRAINT)
opt_policy_1, opt_value_LP, opt_cost_LP = util_methods_1.compute_opt_LP_Constrained()
uncon_policy_1, uncon_value_LP, uncon_cost_LP = util_methods_1.compute_opt_LP_Unconstrained()

print("printing optimal values from LP...")
print(opt_value_LP[0,0])
print("printing optimal values from unconLP...")
print(uncon_value_LP[0,0])
print("printing the optimal costs from LP...")
print(opt_cost_LP[0,0])
# print("printing the unconstrained policy:\n")
# print(uncon_policy_1)
print("\n pringtin the optimal policy from LP",opt_policy_1)


num_samples = 1
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
                BETA[s,a,next_s] = np.sqrt(P_hat[s,a,next_s]*(1-P_hat[s,a,next_s])/num_samples) + 1/(max(num_samples,1))

NUMBER_OF_OCCURANCES = np.full((N_STATES,N_ACTIONS),num_samples)
print(NUMBER_OF_OCCURANCES)


for l in range(NUMBER_SIMULATIONS):
    util_methods = utils(EPS,DELTA,M,P,R,C,EPISODE_LENGTH,N_STATES,N_ACTIONS,CONSTRAINT)
    util_methods.setQvals(opt_value_LP,opt_policy_1)

    VALUE_GAP = np.zeros((NUMBER_SIMULATIONS,NUMBER_EPISODES))
    CONSTRAINT_GAP = np.zeros((NUMBER_SIMULATIONS,NUMBER_EPISODES))
    P_GAP = np.zeros(NUMBER_EPISODES)
    BETA_GAP = np.zeros(NUMBER_EPISODES)
    EP_LENGTH = np.zeros(NUMBER_EPISODES)
    pi_k = np.zeros((N_STATES,EPISODE_LENGTH,N_ACTIONS))

    for k in range(NUMBER_EPISODES):
        if k%50 == 0:
            print("episode running...",k)
        s = 0 
        
        if k == 0:
           util_methods.setEmpiricalKernel(P_hat)
           #util_methods.setConfidenceIntervals(BETA)
           util_methods.setCounts(NUMBER_OF_OCCURANCES_P,NUMBER_OF_OCCURANCES,k)
        
        util_methods.update_empirical_model(k)
        util_methods.compute_confidence_intervals(k)
        
        pi_k, value, cost = util_methods.compute_extended_LP(k)
        #print("--- %s seconds ---" % (time.time() - start_time))
        
           
        VALUE_GAP[l,k] = abs(value[s,0] - opt_value_LP[s,0])
        CONSTRAINT_GAP[l,k] = cost[s,0] - CONSTRAINT
        
        if k == 499:
            print(value[s,0],opt_value_LP[s,0])
            print(cost[s,0],opt_cost_LP[s,0])
            print("printing optimal policies:\n")
            print(opt_policy_1,pi_k)
        
        
       
        episode_count = np.zeros((N_STATES,N_ACTIONS))
        episode_count_p = np.zeros((N_STATES,N_ACTIONS,N_STATES))
        
        
        #Run the episode and collect the information
        for h in range(EPISODE_LENGTH):
            # Step through the episode
            probs = np.zeros(N_ACTIONS)
            for a in range(N_ACTIONS):
                probs[a] = pi_k[s,h,a]
            # print("printing probs:")
            # print(pi_k[s,h,:])
            action = int(np.random.choice(ACTIONS,1,replace=True,p=probs))
            next_state, done = util_methods.step(s,action,h)
            episode_count[s,action]+=1
            episode_count_p[s,action,next_state] += 1
            s = next_state
            
        util_methods.setCounts(episode_count_p,episode_count,k)
        P_GAP[k] = np.linalg.norm(P-util_methods.P_hat)
        BETA_GAP[k] = np.linalg.norm(util_methods.beta_prob)

       
values_gap_mean = np.mean(VALUE_GAP, axis = 0)
cost_gap_mean = np.mean(CONSTRAINT_GAP,axis = 0)
df = pd.DataFrame({'value_gap_mean':values_gap_mean, 'constraint_gap_mean':cost_gap_mean, 'P_GAP':P_GAP, 'BETA_GAP':BETA_GAP, 'EP_LENGTH':EP_LENGTH})
df.to_csv("Values_constrained_LP_FROZENLAKEMDP_test,RunNumber=%d.csv" %(RUN_NUMBER),header=None)

if HPC == False:
    plt.figure(1)
    plt.plot(np.arange(NUMBER_EPISODES),values_gap_mean,'r',label="Objective")
    #plt.plot(np.arange(NUMBER_EPISODES),avg_cum_con_regret,'b',label="Constraint")
    plt.legend(loc="upper left")
    plt.title('Value gap, LP')
    #plt.savefig('RiverswimLP_value_gap.png')
    plt.show()

    plt.figure(2)
    plt.plot(np.arange(NUMBER_EPISODES),cost_gap_mean,'r',label="Objective")
    #plt.plot(np.arange(NUMBER_EPISODES),avg_cum_con_regret,'b',label="Constraint")
    plt.legend(loc="upper left")
    plt.title('cost gap, LP')
    #plt.savefig('Riverswim_constraint_gap.png')
    plt.show()

    plt.figure(2)
    plt.plot(np.arange(NUMBER_EPISODES),P_GAP,'r',label="P norm")
    #plt.plot(np.arange(NUMBER_EPISODES),avg_cum_con_regret,'b',label="Constraint")
    plt.legend(loc="upper left")
    plt.title('Empirical gap, LP')
    #plt.savefig('RiverSwimMDP_value_gap_Phat.png')
    plt.show()

    plt.figure(3)
    plt.plot(np.arange(NUMBER_EPISODES),BETA_GAP,'r',label="Beta Norm")
    #plt.plot(np.arange(NUMBER_EPISODES),avg_cum_con_regret,'b',label="Constraint")
    plt.legend(loc="upper left")
    plt.title('Beta gap, LP')
    #plt.savefig('Riverswim_value_gap_Phat.png')
    plt.show()

print("--- %s seconds ---" % (time.time() - start_time))

