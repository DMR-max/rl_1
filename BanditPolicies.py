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
        self.action_tried = np.zeros(n_actions)
        self.action_val = np.zeros(n_actions)
        self.action_mean = np.zeros(n_actions)
        
    def select_action(self, epsilon):
        random_float = np.random.random()
        if random_float < (1 - epsilon):
            
            a = np.argmax(self.action_val)

        else:
            # kies argmax van de lijst met actie waarden
            # kies een random actie 
            # haal beste actie eruit
            b = np.argmax(self.action_val)
            probability = []
            for i in range(self.n_actions):
                probability.append(i)
                
            probability.remove(b)
            a = int(np.random.choice(probability,size = 1))

        return a
        
    def update(self,a,r):
        self.action_tried[a] += 1
        
        value = (r - self.action_val[a])
        value2 = self.action_tried[a]
        self.action_val[a] += (value / value2)


class OIPolicy:

    def __init__(self, n_actions=10, initial_value=0.0, learning_rate=0.1):
        self.n_actions = n_actions
        self.est_action_val = np.full(n_actions, initial_value)
        self.learning_rate = learning_rate
        self.est_action_val = np.full(n_actions, initial_value)
        self.learning_rate = learning_rate
        
    def select_action(self):

        a = np.argmax(self.est_action_val)
        return a
        
    def update(self,a,r):
        
        value = r - self.est_action_val[a]

        self.est_action_val[a] = self.est_action_val[a] + (self.learning_rate * value)


class UCBPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        self.action_tried = np.zeros(n_actions)
        self.action_val = np.zeros(n_actions)
        pass
    
    def select_action(self, c, t):
        action_val_with_upperconf = np.zeros(self.n_actions)
        for i in range(self.n_actions):
            if self.action_tried[i] == 0:
                action_val_with_upperconf[i] = float('inf')
            else:
                action_val_with_upperconf[i] = self.action_val[i] + c * (np.sqrt(np.log(t) / (self.action_tried[i])))
        a = np.argmax(action_val_with_upperconf)
        return a
        
    def update(self,a,r):
        self.action_tried[a] += 1
        
        value = (r - self.action_val[a])
        value2 = self.action_tried[a]

        self.action_val[a] += (value / value2)
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
