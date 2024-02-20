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
from BanditPolicies import EgreedyPolicy, OIPolicy, UCBPolicy
from Helper import LearningCurvePlot, ComparisonPlot, smooth
 

def experiment(n_actions, n_timesteps, n_repetitions, smoothing_window):
    #To Do: Write all your experiment code here
    average_reward_list = []
    plot = LearningCurvePlot("plot Epsilon Greedy") # For da plot
    for j in range(n_repetitions):
        reward_list = []
        env = BanditEnvironment(n_actions=n_actions) # Initialize environment 
        pi = EgreedyPolicy(n_actions=n_actions) # Initialize policy
        # print("Mean pay-off per action: {}".format(env.means))
        # print("Best action = {} with mean pay-off {}".format(env.best_action,env.best_average_return))

        for i in range(n_timesteps):
            a = pi.select_action(epsilon=0.25) # select action
            r = env.act(a) # sample reward
            pi.update(a,r) # update policy
            reward_list.append(r)
            # print("Test e-greedy policy with action {}, received reward {}".format(a,r))
        average_this_rep = sum(reward_list) / len(reward_list)
        average_reward_list.append(average_this_rep)

        # print(pi.action_val) # printing final values this repetition
    vector_reward = np.array(average_reward_list) # for curve
    vector_reward_smoothed = smooth(vector_reward, smoothing_window)
    plot.add_curve(vector_reward_smoothed)
    plot.save()
    
    # Assignment 1: e-greedy
    
    # Assignment 2: Optimistic init
    
    # Assignment 3: UCB
    
    # Assignment 4: Comparison
    
    pass

if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31
    
    experiment(n_actions=n_actions,n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window)