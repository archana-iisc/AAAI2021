#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:43:50 2020

@author: archu
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:43:45 2020

@author: Archana
"""
import numpy as np
import pulp as p
import time
import math
import sys

class utils:
    def __init__(self,eps, delta, M, P,R,C,EPISODE_LENGTH,N_STATES,N_ACTIONS,CONSTRAINT):
        self.P = P
        self.R = R
        self.C = C
        self.EPISODE_LENGTH = EPISODE_LENGTH
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
        self.eps = eps
        self.delta = delta
        self.M = M
        self.ENV_Q_VALUES = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_ACTIONS))
        
        self.P_hat = np.zeros((self.N_STATES,self.N_ACTIONS,self.N_STATES))
        
        
            
        self.R_hat = np.zeros((self.N_STATES,self.N_ACTIONS))
        self.C_hat = np.zeros((self.N_STATES,self.N_ACTIONS))
        self.R_tilde = np.zeros((self.N_STATES,self.N_ACTIONS))
        self.C_tilde = np.zeros((self.N_STATES,self.N_ACTIONS))
        
        
        self.NUMBER_OF_OCCURANCES = np.zeros((self.N_STATES,self.N_ACTIONS)) 
        self.NUMBER_OF_OCCURANCES_p = np.zeros((self.N_STATES,self.N_ACTIONS,self.N_STATES)) 
        
        
        for s in range(self.N_STATES):
          for a in range(self.N_ACTIONS):
            #self.P_hat[s,a,:] = 1/self.N_STATES
            #self.NUMBER_OF_OCCURANCES[s,a] = 1
            #self.NUMBER_OF_OCCURANCES_p[s,a,s] = 1
            
            self.NUMBER_OF_OCCURANCES[s,a] = 0
            self.NUMBER_OF_OCCURANCES_p[s,a,s] = 0
            
            
        #print(np.sum(self.P_hat,axis=2))

        #Optimism
        self.beta_cost = np.zeros((self.N_STATES,self.N_ACTIONS))
        self.beta_prob = np.zeros((self.N_STATES,self.N_ACTIONS,self.N_STATES))
        
        self.beta_cost_1 = np.zeros((self.N_STATES,self.N_ACTIONS))
        self.beta_prob_1 = np.zeros((self.N_STATES,self.N_ACTIONS))
        
        self.STATES = np.arange(self.N_STATES)
        self.ACTIONS = np.arange(self.N_ACTIONS)
        #self.mu = np.full(self.N_STATES,1/self.N_STATES)
        self.mu = np.zeros(self.N_STATES)
        self.mu[0] = 1.0
        self.CONSTRAINT = CONSTRAINT
        
        self.total_reward =  np.zeros((self.N_STATES,self.N_ACTIONS))
        self.total_cost =  np.zeros((self.N_STATES,self.N_ACTIONS))
        
        self.qVals = {}
        self.qMax = {}
        
        self.OPT_POLICY = np.zeros((self.N_STATES,self.EPISODE_LENGTH))
        self.opt_policy_prev = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_ACTIONS))
        
    def setQvals(self,qvals,opt_policy):
        #self.ENV_Q_VALUES = qvals
        self.OPT_POLICY = opt_policy 
    
    def setEmpiricalKernel(self,P_hat):
        self.P_hat = P_hat
        return
    
    def setConfidenceIntervals(self,beta):
        self.beta_prob = beta
        return
        
    def step(self,s, a, h):
        pContinue = 0
        probs = np.zeros((self.N_STATES))
        for next_s in range(self.N_STATES):
            probs[next_s] = self.P[s,a,next_s]
        next_state = int(np.random.choice(self.STATES,1,replace=True,p=probs))
        #reward = self.R[s,a]
        #cost = self.C[s,a]
        Done_Indices = [5,7,11,12,15]
        Done_Indices_1 = [8]
        if next_state in Done_Indices_1:
            pContinue = 1
        return next_state, pContinue
    
    def setCounts(self,ep_count_p,ep_count,ep):
        # if ep == 0:
        #     for s in range(self.N_STATES):
        #      for a in range(self.N_ACTIONS):
        #          self.NUMBER_OF_OCCURANCES[s,a] = self.N_STATES
        #          for s_1 in range(self.N_STATES):
        #              self.NUMBER_OF_OCCURANCES_p[s,a,s_1] = 1
        self.NUMBER_OF_OCCURANCES_p += ep_count_p
        self.NUMBER_OF_OCCURANCES += ep_count
    
    
    def compute_confidence_intervals(self,ep):
        #self.beta_prob = (2/np.sqrt(np.maximum(self.NUMBER_OF_OCCURANCES,1)))
     
        for s in range(self.N_STATES):
            for a in range(self.N_ACTIONS):
                for s_1 in range(self.N_STATES):
                    self.beta_prob[s,a,s_1] = min(np.sqrt(self.P_hat[s,a,s_1]*(1-self.P_hat[s,a,s_1])/max(self.NUMBER_OF_OCCURANCES[s,a],1)) + 1/(max(self.NUMBER_OF_OCCURANCES[s,a],1)), 1/(max(np.sqrt(self.NUMBER_OF_OCCURANCES[s,a]),1)))
        return
    
    def update_empirical_model_old(self):
        for s in range(self.N_STATES):
              for a in range(self.N_ACTIONS):
                for s_1 in range(self.N_STATES):
                  self.P_hat[s,a,s_1] = self.NUMBER_OF_OCCURANCES_p[s,a,s_1]/(max(self.NUMBER_OF_OCCURANCES[s,a],1))
                self.P_hat[s,a,:] /= np.sum(self.P_hat[s,a,:])
                if sum(self.P_hat[s,a,:]) != 1:
                    print("empirical is wrong")
                
        return 
    
    def update_empirical_model(self,ep):
        for s in range(self.N_STATES):
              for a in range(self.N_ACTIONS):
                for s_1 in range(self.N_STATES):
                  # if ep == 0:
                  #     self.P_hat[s,a,:] = 0
                  #     #next_s = min(s+1,self.N_STATES-1)
                  #     next_s = s
                  #     self.P_hat[s,a,next_s] = 1
                  #     self.NUMBER_OF_OCCURANCES_p[s,a,next_s] = 1
                  #     self.NUMBER_OF_OCCURANCES[s,a] = 1
                  # else:
                  self.P_hat[s,a,s_1] = self.NUMBER_OF_OCCURANCES_p[s,a,s_1]/(max(self.NUMBER_OF_OCCURANCES[s,a],1))
                self.P_hat[s,a,:] /= np.sum(self.P_hat[s,a,:])
                if abs(sum(self.P_hat[s,a,:]) - 1)  >  0.001:
                    print("empirical is wrong")
                    print(self.P_hat)
                
        return
    
    def compute_opt_LP_Unconstrained(self):
        opt_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_ACTIONS))
        optimal_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH))
        opt_prob = p.LpProblem("OPT LP problem",p.LpMaximize)
        opt_q = np.zeros((self.EPISODE_LENGTH,self.N_STATES,self.N_ACTIONS))
        
        #create problem variables
        q_keys = [(h,s,a) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) 
                                 for a in range(self.N_ACTIONS)]
        q = p.LpVariable.dicts("q",q_keys,lowBound=0,cat='Continuous')
        
        #Objective function
        list_1 = [self.R[h,s,a] for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in range(self.N_ACTIONS)] * self.EPISODE_LENGTH
        list_2 = [q[(h,s,a)] for  h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) 
                                  for a in range(self.N_ACTIONS)]
        opt_prob += p.lpDot(list_1,list_2)
        
        #opt_prob += p.lpSum([q[(h,s,a)]*self.R[s,a] for  h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) 
                                #for a in range(self.N_ACTIONS)]) 
        
        for h in range(1,self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                q_list = [q[(h,s,a)] for a in range(self.N_ACTIONS)]
                pq_list = [self.P[s_1,a_1,s]*q[(h-1,s_1,a_1)] for s_1 in range(self.N_STATES) for a_1 in range(self.N_ACTIONS)]
                opt_prob += p.lpSum(q_list) - p.lpSum(pq_list) == 0
        for s in range(self.N_STATES):
            q_list = [q[(0,s,a)] for a in range(self.N_ACTIONS)]
            opt_prob += p.lpSum(q_list) - self.mu[s] == 0
            
        # for h in range(self.EPISODE_LENGTH):
        #         opt_prob += p.lpSum([q[(h,s,a)] for s in range(self.N_STATES) for a in range(self.N_ACTIONS)]) - 1 == 0
    
        status = opt_prob.solve(p.PULP_CBC_CMD(fracGap=0.001))
        #print(p.LpStatus[status])   # The solution status 
        #print(opt_prob)
        # print("printing best value")
        # print(p.value(opt_prob.objective))
        
        # for constraint in opt_prob.constraints:
        #     print(opt_prob.constraints[constraint].name, opt_prob.constraints[constraint].value() - opt_prob.constraints[constraint].constant)
        
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in range(self.N_ACTIONS):
                    opt_q[h,s,a] = q[(h,s,a)].varValue
                for a in range(self.N_ACTIONS):
                    if np.sum(opt_q[h,s,:]) == 0:
                        opt_policy[s,h,a] = 1/self.N_ACTIONS
                    else:
                        opt_policy[s,h,a] = opt_q[h,s,a]/np.sum(opt_q[h,s,:])
                probs = opt_policy[s,h,:]
                    
                optimal_policy[s,h] = int(np.argmax(probs))
        
        value_of_policy, cost_of_policy = self.FiniteHorizon_Policy_evaluation(opt_policy)
        
        return opt_policy, value_of_policy, cost_of_policy
    
    def compute_opt_LP_Constrained(self):
        
        opt_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_ACTIONS))
        optimal_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH))
        opt_prob = p.LpProblem("OPT LP problem",p.LpMaximize)
        opt_q = np.zeros((self.EPISODE_LENGTH,self.N_STATES,self.N_ACTIONS))
        
        #create problem variables
        q_keys = [(h,s,a) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) 
                                 for a in range(self.N_ACTIONS)]
        
        q = p.LpVariable.dicts("q",q_keys,lowBound=0,cat='Continuous')
        
        #Objective function
        #list_1 = [self.R[h,s,a] for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in range(self.N_ACTIONS)] * self.EPISODE_LENGTH
        #list_2 = [q[(h,s,a)] for  h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) 
                                  #for a in range(self.N_ACTIONS)]
        #opt_prob += p.lpDot(list_1,list_2)
        
        opt_prob += p.lpSum([q[(h,s,a)]*self.R[h,s,a] for  h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) 
                                for a in range(self.N_ACTIONS)]) 
        
        #Constraints
        #list_1 = [self.C[h,s,a] for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in range(self.N_ACTIONS)] * self.EPISODE_LENGTH
        #list_2 = [q[(h,s,a)] for  h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) 
                                  #for a in range(self.N_ACTIONS)]
        #opt_prob += p.lpDot(list_1,list_2) - self.CONSTRAINT <= 0
        opt_prob += p.lpSum([q[(h,s,a)]*self.C[h,s,a] for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in range(self.N_ACTIONS)]) - self.CONSTRAINT <= 0
        print("hello")
        
        for h in range(1,self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                q_list = [q[(h,s,a)] for a in range(self.N_ACTIONS)]
                pq_list = [self.P[s_1,a_1,s]*q[(h-1,s_1,a_1)] for s_1 in range(self.N_STATES) for a_1 in range(self.N_ACTIONS)]
                opt_prob += p.lpSum(q_list) - p.lpSum(pq_list) == 0
        for s in range(self.N_STATES):
            q_list = [q[(0,s,a)] for a in range(self.N_ACTIONS)]
            opt_prob += p.lpSum(q_list) - self.mu[s] == 0
            
        # for h in range(self.EPISODE_LENGTH):
        #         opt_prob += p.lpSum([q[(h,s,a)] for s in range(self.N_STATES) for a in range(self.N_ACTIONS)]) - 1 == 0
    
        status = opt_prob.solve(p.PULP_CBC_CMD(fracGap=0.001))
        #print(p.LpStatus[status])   # The solution status 
        #print(opt_prob)
        # print("printing best value")
        # print(p.value(opt_prob.objective))
        
        # for constraint in opt_prob.constraints:
        #     print(opt_prob.constraints[constraint].name, opt_prob.constraints[constraint].value() - opt_prob.constraints[constraint].constant)
        
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in range(self.N_ACTIONS):
                    opt_q[h,s,a] = q[(h,s,a)].varValue
                for a in range(self.N_ACTIONS):
                    if np.sum(opt_q[h,s,:]) == 0:
                        opt_policy[s,h,a] = 1/self.N_ACTIONS
                    else:
                        opt_policy[s,h,a] = opt_q[h,s,a]/np.sum(opt_q[h,s,:])
                        if math.isnan(opt_policy[s,h,a]):
                            opt_policy[s,h,a] = 1/self.N_ACTIONS
                        elif opt_policy[s,h,a] > 1.0:
                            print("invalid value printing")
                            print(opt_policy[s,h,a])
                #probs = opt_policy[s,h,:]
                    
                #optimal_policy[s,h] = int(np.argmax(probs))
       
        value_of_policy, cost_of_policy = self.FiniteHorizon_Policy_evaluation(opt_policy)
        
        return opt_policy, value_of_policy, cost_of_policy
    
    def compute_extended_LP(self,ep):
        
        opt_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_ACTIONS))
        optimal_policy =  np.zeros((self.N_STATES,self.EPISODE_LENGTH))
        opt_prob = p.LpProblem("OPT LP problem",p.LpMaximize)
        opt_z = np.zeros((self.EPISODE_LENGTH,self.N_STATES,self.N_ACTIONS,self.N_STATES))
        
        #create problem variables
        z_keys = [(h,s,a,s_1) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) 
                                 for a in range(self.N_ACTIONS) for s_1 in range(self.N_STATES)]
        
        
        z = p.LpVariable.dicts("z",z_keys,lowBound=0,upBound=1,cat='Continuous')
        
        
        #Objective function
        opt_prob += p.lpSum([z[(h,s,a,s_1)]*self.R[h,s,a] for  h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) 
                                 for a in range(self.N_ACTIONS) for s_1 in range(self.N_STATES)]) 
        
        #Constraints
        opt_prob += p.lpSum([z[(h,s,a,s_1)]*self.C[h,s,a] for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in range(self.N_ACTIONS) for s_1 in range(self.N_STATES)]) - self.CONSTRAINT <= 0
        
        
        for h in range(1,self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                z_list = [z[(h,s,a,s_1)] for a in range(self.N_ACTIONS) for s_1 in range(self.N_STATES)]
                z_1_list = [z[(h-1,s_1,a_1,s)] for s_1 in range(self.N_STATES) for a_1 in range(self.N_ACTIONS)]
                opt_prob += p.lpSum(z_list) - p.lpSum(z_1_list) == 0
        
        for s in range(self.N_STATES):
            q_list = [z[(0,s,a,s_1)] for a in range(self.N_ACTIONS) for s_1 in range(self.N_STATES)]
            opt_prob += p.lpSum(q_list) - self.mu[s] == 0
        
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
              for a in range(self.N_ACTIONS):
                  for s_1 in range(self.N_STATES):
                       opt_prob += z[(h,s,a,s_1)] - (self.P_hat[s,a,s_1] + self.beta_prob[s,a,s_1]) *  p.lpSum([z[(h,s,a,y)] for y in range(self.N_STATES)]) <= 0
                       opt_prob += -z[(h,s,a,s_1)] + (self.P_hat[s,a,s_1] - self.beta_prob[s,a,s_1])* p.lpSum([z[(h,s,a,y)] for y in range(self.N_STATES)]) <= 0
                       #opt_prob += z[(h,s,a,s_1)] - (self.P[s,a,s_1]) *  p.lpSum([z[(h,s,a,y)] for y in range(self.N_STATES)]) <= 0
                       #opt_prob += -z[(h,s,a,s_1)] + (self.P[s,a,s_1])* p.lpSum([z[(h,s,a,y)] for y in range(self.N_STATES)]) <= 0
        
        #status = opt_prob.solve(p.GLPK(msg = 0))
       
        status = opt_prob.solve(p.PULP_CBC_CMD(fracGap=0.01))
        
        #print("Computing extended LP...\n")
        #print(p.LpStatus[status])   # The solution status 
        
         
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in range(self.N_ACTIONS):
                  for s_1 in range(self.N_STATES):
                    opt_z[h,s,a,s_1] = z[(h,s,a,s_1)].varValue
                    if opt_z[h,s,a,s_1] < 0 and opt_z[h,s,a,s_1] > -0.0001:
                        opt_z[h,s,a,s_1] = 0
                        
                    elif opt_z[h,s,a,s_1] < -0.0001:
                        print("invalid value")
                        sys.exit()
        
       
         
        
        den = np.sum(opt_z,axis=(2,3))
        num = np.sum(opt_z,axis=3)
        
        
                  
        
        
        
        # den = np.zeros((self.EPISODE_LENGTH,self.N_STATES))
        # num = np.zeros((self.EPISODE_LENGTH,self.N_STATES,self.N_ACTIONS))
        # for h in range(self.EPISODE_LENGTH):
        #     for s in range(self.N_STATES):
        #         den[h,s] = 0
        #         for a in range(self.N_ACTIONS):
        #             for s_1 in range(self.N_STATES):
        #                 den[h,s] += opt_z[h,s,a,s_1]
                      
        # for h in range(self.EPISODE_LENGTH):
        #     for s in range(self.N_STATES):
        #       for a in range(self.N_ACTIONS):
        #           num[h,s,a] = 0
        #           for s_1 in range(self.N_STATES):
        #             num[h,s,a] += opt_z[h,s,a,s_1]
        
        # print(den)
        # print(num)
        #print(den.shape,num.shape)
         
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                 #print(den[h,s])
                # if den[h,s] == 0:
                #         continue
                #         opt_policy[s,h,:] = 1/self.N_ACTIONS
                sum_prob = 0
                for a in range(self.N_ACTIONS):
                        opt_policy[s,h,a] = num[h,s,a]/den[h,s]
                        # if opt_policy[s,h,a] > 1.0:
                        #     # print("invalid value")
                        #     # print(opt_policy[s,h,a])
                        #     opt_policy[s,h,a] = 1.0
                            
                        # elif opt_policy[s,h,a] < 0:
                        #     print("invalid value for policy")
                        #     print(opt_policy[s,h,a])
                        #     sys.exit()
                        #     opt_policy[s,h,a] = 0
                        sum_prob += opt_policy[s,h,a]
                if abs(sum(num[h,s,:]) - den[h,s]) > 0.0001:
                    print("wrong values")
                    print(sum(num[h,s,:]),den[h,s])
                    sys.exit()
                
                        
                # if sum_prob == 0:
                #         #continue
                #         opt_policy[s,h,:] = 1/self.N_ACTIONS
                #         #opt_policy[s,h,:] = 0
                #         #opt_policy[s,h,0] = 0.5
                #         #opt_policy[s,h,1] = 0.5
                if math.isnan(sum_prob):
                        # print("Nan Exception")
                        # sys.exit()
                        # opt_policy[s,h,:] = 0
                        # opt_policy[s,h,0] = 0.5
                        # opt_policy[s,h,1] = 0.5
                    #opt_policy[s,h,:] = self.opt_policy_prev[s,h,:]   
                    #if ep == 0:   
                     opt_policy[s,h,:] = 1/self.N_ACTIONS
                     if ep == 45:
                         print("state")
                         print(h,s)
                     
                # elif sum_prob > 1.0:
                #     opt_policy[s,h,:] = 1/self.N_ACTIONS
                
                # elif sum_prob < 0.0:
                #     opt_policy[s,h,:] = 1/self.N_ACTIONS
                     
                else:
                        for a in range(self.N_ACTIONS):
                            opt_policy[s,h,a] = opt_policy[s,h,a]/sum_prob
                #self.opt_policy_prev[s,h,:] = opt_policy[s,h,:]
                #sum_prob = np.sum(opt_policy,axis = 2)
                
                            
        if ep == 0:
            print("printing optimal policy from ext LP")
            print(opt_policy)
        
                
        value_of_policy, cost_of_policy = self.FiniteHorizon_Policy_evaluation(opt_policy)
        
        
        
        return opt_policy, value_of_policy, cost_of_policy
    
    
    #Evaluate the policy to obtain the value
    def FiniteHorizon_Policy_evaluation_deterministic(self,P,policy):
        v = np.zeros((self.N_STATES,self.EPISODE_LENGTH))
        c = np.zeros((self.N_STATES,self.EPISODE_LENGTH))
        for s in range(self.N_STATES):
            a = int(policy[s,self.EPISODE_LENGTH-1])
            v[s,self.EPISODE_LENGTH-1] = self.R[s,a]
            c[s,self.EPISODE_LENGTH-1] = self.C[s,a]
            
        for h in range(self.EPISODE_LENGTH-2,-1,-1):
          for s in range(self.N_STATES):
            v[s,h] = self.R[s,int(policy[s,h])] + np.dot(P[s,int(policy[s,h]),:],v[:,h+1])
            c[s,h] = self.C[s,int(policy[s,h])] + np.dot(P[s,int(policy[s,h]),:],c[:,h+1])
        return v, c
    
    def FiniteHorizon_Policy_evaluation(self,policy):
        v = np.zeros((self.N_STATES,self.EPISODE_LENGTH))
        c = np.zeros((self.N_STATES,self.EPISODE_LENGTH))
        P_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_STATES))
        R_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH))
        C_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH))
        for s in range(self.N_STATES):
            v[s,self.EPISODE_LENGTH-1] = np.dot(policy[s,self.EPISODE_LENGTH-1,:],self.R[self.EPISODE_LENGTH-1,s,:])
            c[s,self.EPISODE_LENGTH-1] = np.dot(policy[s,self.EPISODE_LENGTH-1,:],self.C[self.EPISODE_LENGTH-1,s,:])
        for h in range(self.EPISODE_LENGTH):   
            for s in range(self.N_STATES):
                R_policy[s,h] = np.dot(policy[s,h,:],self.R[h,s,:])
                C_policy[s,h] = np.dot(policy[s,h,:],self.C[h,s,:])
                for s_1 in range(self.N_STATES):
                    P_policy[s,h,s_1] = np.dot(policy[s,h,:],self.P[s,:,s_1])
                    
        for h in range(self.EPISODE_LENGTH-2,-1,-1):
          for s in range(self.N_STATES):
                 v[s,h] = R_policy[s,h] + np.dot(P_policy[s,h,:],v[:,h+1])
                 c[s,h] = C_policy[s,h] + np.dot(P_policy[s,h,:],c[:,h+1])
        return v, c
    
    def FiniteHorizon_Optimal_Backward_Recursion(self):
        
        v = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_ACTIONS))
        v_1 = np.zeros((self.N_STATES,self.EPISODE_LENGTH))
        policy_1 = np.zeros((self.N_STATES,self.EPISODE_LENGTH))
        
        for s in range(self.N_STATES):
            for a in range(self.N_ACTIONS):
                v[s,self.EPISODE_LENGTH-1,a] = self.R[self.EPISODE_LENGTH-1,s,a]
            
            v_1[s,self.EPISODE_LENGTH-1] = np.max(v[s,self.EPISODE_LENGTH-1,:])
            policy_1[s,self.EPISODE_LENGTH-1] = np.argmax(v[s,self.EPISODE_LENGTH-1,:])
         
        for h in range(self.EPISODE_LENGTH-2,-1,-1):
          for s in range(self.N_STATES):
            for a in range(self.N_ACTIONS):
                v[s,h,a] = self.R[h,s,a]
                for s_1 in range(self.N_STATES):
                    v[s,h,a] += self.P[s,a,s_1]*v_1[s_1,h+1]
            v_1[s,h] = np.max(v[s,h,:])
            policy_1[s,h] = np.argmax(v[s,h,:]) 
        self.ENV_Q_VALUES = v
            
        return v, policy_1, v_1
    
    def compute_qVals_EVI(self):
        '''
        Compute the Q values for a given R, P by extended value iteration

        Args:
            R - R[s,a] = mean rewards
            P - P[s,a] = probability vector of transitions
            R_slack - R_slack[s,a] = slack for rewards
            P_slack - P_slack[s,a] = slack for transitions

        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        '''
                # Extended value iteration
        qVals = {}
        qMax = {}
        qMax[self.EPISODE_LENGTH] = np.zeros(self.N_STATES)

        for h in range(self.EPISODE_LENGTH):
            j = self.EPISODE_LENGTH - h - 1
            qMax[j] = np.zeros(self.N_STATES)

            for s in range(self.N_STATES):
                qVals[s, j] = np.zeros(self.N_ACTIONS)

                for a in range(self.N_ACTIONS):
                    #rOpt = R[s, a] + R_slack[s, a]

                    # form pOpt by extended value iteration, pInd sorts the values
                    pInd = np.argsort(qMax[j + 1])
                    pOpt = self.P[s, a]
                    if pOpt[pInd[self.N_STATES - 1]] + self.beta_prob1[s, a] * 0.5 > 1:
                        pOpt = np.zeros(self.N_STATES)
                        pOpt[pInd[self.N_STATES - 1]] = 1
                    else:
                        pOpt[pInd[self.N_STATES - 1]] += self.beta_prob1[s, a] * 0.5

                    # Go through all the states and get back to make pOpt a real prob
                    sLoop = 0
                    while np.sum(pOpt) > 1:
                        worst = pInd[sLoop]
                        pOpt[worst] = max(0, 1 - np.sum(pOpt) + pOpt[worst])
                        sLoop += 1

                    # Do Bellman backups with the optimistic R and P
                    qVals[s, j][a] = self.R[s,a] + np.dot(pOpt, qMax[j + 1])
                    #print(pOpt)

                qMax[j][s] = np.max(qVals[s, j])

        return qVals, qMax
    
    def update_policy_from_EVI(self):
        '''
        Compute UCRL2 Q-values via extended value iteration.
        '''
        # Perform extended value iteration
        qVals, qMax = self.compute_qVals_EVI()
        policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH))
        
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                Q = qVals[s,h]
                policy[s,h] = np.random.choice(np.where(Q==Q.max())[0])

        self.qVals = qVals
        self.qMax = qMax
        return policy
    
    
    def compute_qVals(self):
        '''
        Compute the Q values for the environment

        Args:
            NULL - works on the TabularMDP

        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        '''
        qVals = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_ACTIONS))
        qMax = np.zeros((self.N_STATES,self.EPISODE_LENGTH+1))

        for i in range(self.EPISODE_LENGTH):
            j = self.EPISODE_LENGTH - i - 1

            for s in range(self.N_STATES):
                for a in range(self.N_ACTIONS):
                    qVals[s, j, a] = self.R[s, a] + float(np.dot(self.P[s, a,:], qMax[:,j + 1]))

                qMax[s,j] = float(np.amax(qVals[s, j,:]))
        self.ENV_Q_VALUES = qVals
        # print("printing qvals shape")
        # print(self.ENV_Q_VALUES)
        return qVals, qMax
    
    
