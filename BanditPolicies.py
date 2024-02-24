#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit environment
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
from BanditEnvironment import BanditEnvironment

class EgreedyPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        self.action_val = np.zeros(n_actions)
        # TO DO: Add own code
        pass
        
    def select_action(self, epsilon):
        random_float = np.random.uniform(0,1) 
        if random_float < epsilon:
            # kies een random actie
            a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        else:
            # kies argmax van de lijst met actie waarden
            a = np.argmax(self.action_val)
        
        return a
        
    def update(self,a,r):
        value = (r-self.action_val[a]) / (self.n_actions + 1)
        self.action_val[a] += value
        pass

class OIPolicy:

    def __init__(self, n_actions=10, initial_value=0.0, learning_rate=0.1):
        self.n_actions = n_actions
        self.est_action_val = np.full(n_actions, initial_value)
        self.learning_rate = learning_rate
        
    def select_action(self):
        # TO DO: Add own code
        #a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        # print("est_list: ", self.est_action_val)
        a = np.argmax(self.est_action_val)
        return a
        
    def update(self,a,r):
        # print("r: ", r)
        value = r - self.est_action_val[a]
        # print("r-old: ", value)
        # print("old_value: ", self.est_action_val[a])
        self.est_action_val[a] = self.est_action_val[a] + (self.learning_rate * value)
        # print("new_value: ", self.est_action_val[a])
        # print('est list')
        # print(self.est_action_val)

class UCBPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        # TO DO: Add own code
        pass
    
    def select_action(self, c, t):
        # TO DO: Add own code
        a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        return a
        
    def update(self,a,r):
        # TO DO: Add own code
        pass
    
def test():
    n_actions = 10
    env = BanditEnvironment(n_actions=n_actions) # Initialize environment 
    pi = EgreedyPolicy(n_actions=n_actions) # Initialize policy   
    print("Mean pay-off per action: {}".format(env.means))
    print("Best action = {} with mean pay-off {}".format(env.best_action,env.best_average_return))

    for i in range(10):
        a = pi.select_action(epsilon=0.5) # select action
        r = env.act(a) # sample reward
        pi.update(a,r) # update policy
        print("Test e-greedy policy with action {}, received reward {}".format(a,r))
    
    pi = OIPolicy(n_actions=n_actions,initial_value=1.0) # Initialize policy
    a = pi.select_action() # select action
    r = env.act(a) # sample reward
    pi.update(a,r) # update policy
    print("Test greedy optimistic initialization policy with action {}, received reward {}".format(a,r))
    
    pi = UCBPolicy(n_actions=n_actions) # Initialize policy
    a = pi.select_action(c=1.0,t=1) # select action
    r = env.act(a) # sample reward
    pi.update(a,r) # update policy
    print("Test UCB policy with action {}, received reward {}".format(a,r))
    
if __name__ == '__main__':
    test()
