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
        # list with number of times each action is used
        self.action_tried = np.zeros(n_actions)
        # list with the value of each action
        self.action_val = np.zeros(n_actions)
        self.action_mean = np.zeros(n_actions)
        
    def select_action(self, epsilon):
        random_float = np.random.random()
        if random_float < (1 - epsilon):
            
            a = np.argmax(self.action_val)

        else:
            # choose argmax from list of action values
            b = np.argmax(self.action_val)
            # take out the best action from list of actions
            probability = []
            for i in range(self.n_actions):
                probability.append(i)
                
            probability.remove(b)
            # choose a random action from the list of actions
            a = int(np.random.choice(probability,size = 1))

        return a
        
    def update(self,a,r):
        # update the counter per action a in a list of actions
        self.action_tried[a] += 1
        
        # update the value of action a in a list of action values
        value = (r - self.action_val[a])
        value2 = self.action_tried[a]
        self.action_val[a] += (value / value2)


class OIPolicy:

    def __init__(self, n_actions=10, initial_value=0.0, learning_rate=0.1):
        self.n_actions = n_actions
        # list with the inital value of each action
        self.est_action_val = np.full(n_actions, initial_value)
        # learning rate
        self.learning_rate = learning_rate
        
    def select_action(self):
        # choose the best action from the list of action values
        a = np.argmax(self.est_action_val)
        return a
        
    def update(self,a,r):
        # get the last part of the equation
        value = r - self.est_action_val[a]
        # update the list with the correct values for action a
        self.est_action_val[a] = self.est_action_val[a] + (self.learning_rate * value)


class UCBPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        # list with number of times each action is used
        self.action_tried = np.zeros(n_actions)
        # list with the value of each action
        self.action_val = np.zeros(n_actions)
    
    def select_action(self, c, t):
        # create a list with the UCB (upper confidence bounds) for each action
        action_val_with_upperconf = np.zeros(self.n_actions)
        # for loop for calculating the UCB for each action
        for i in range(self.n_actions):
            # if the action hasn't been accessed yet then the UCB is infinite
            if self.action_tried[i] == 0:
                action_val_with_upperconf[i] = float('inf')
            else:
                # calculate the UCB for each action that is already accessed
                action_val_with_upperconf[i] = self.action_val[i] + c * (np.sqrt(np.log(t) / (self.action_tried[i])))
        # choose the argmax from the list of calculated UCB's 
        a = np.argmax(action_val_with_upperconf)
        return a
        
    def update(self,a,r):
        # update the counter per action a in a list of actions
        self.action_tried[a] += 1
        
        # numerator of the equation
        value = (r - self.action_val[a])
        # denominator of the equation
        value2 = self.action_tried[a]
        # update the value of action a in a list of action values
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
