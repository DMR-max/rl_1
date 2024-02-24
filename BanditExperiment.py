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
import matplotlib.pyplot as plt

 

def experiment(n_actions, n_timesteps, n_repetitions, smoothing_window, assignment):
    #To Do: Write all your experiment code here
    average_reward_list = []

    if assignment == 1:
        plot = LearningCurvePlot("plot Epsilon Greedy") # For da plot
    elif assignment == 2:
        plot = LearningCurvePlot("plot Optimistic Initialization") # For da plot
    elif assignment == 3:
        plot = LearningCurvePlot("plot Upper Confidence Bounds") # For da plot

    for j in range(n_repetitions):
        optimal_action_counter = 0
        reward_list = []
        env = BanditEnvironment(n_actions=n_actions) # Initialize environment
        if assignment == 1:
            pi = EgreedyPolicy(n_actions=n_actions) # Initialize policy
        elif assignment == 2:
            pi = OIPolicy(n_actions=n_actions, initial_value=0.8)
        elif assignment == 3:
            pi = UCBPolicy(n_actions=n_actions)
        # print("Mean pay-off per action: {}".format(env.means))
        # print("Best action = {} with mean pay-off {}".format(env.best_action,env.best_average_return))

        for i in range(n_timesteps):
            if assignment == 1:
                a = pi.select_action(epsilon=0.25) # select action
            else:
                a = pi.select_action() # select action
            r = env.act(a) # sample reward
            pi.update(a,r) # update policy
            reward_list.append(r)
            # print("best_action: ", env.best_action)
            # print("a: ", a)
            optimal_action_counter += int(a == env.best_action)
            # print("optimal_action_counter_updated: ", optimal_action_counter)
            # print("Test e-greedy policy with action {}, received reward {}".format(a,r))
        
        if assignment == 2:
            # print("optimal_action_counter: ", optimal_action_counter)
            # print("n_timesteps: ", n_timesteps)
            average_optimal_action_percentage = (optimal_action_counter / n_timesteps)
            average_reward_list.append(average_optimal_action_percentage)
        else:
            average_this_rep = sum(reward_list) / len(reward_list)
            average_reward_list.append(average_this_rep)

        # print(pi.action_val) # printing final values this repetition
    vector_reward = np.array(average_reward_list) # for curve
    vector_reward_smoothed = smooth(vector_reward, smoothing_window)

    if assignment == 1:
        plot.add_curve(vector_reward_smoothed, label="Epsilon Greedy")
    elif assignment == 2:
        plot.add_curve(vector_reward_smoothed, label="Optimistic Initialization")
    elif assignment == 3:
        plot.add_curve(vector_reward_smoothed, label="Upper Confidence Bounds")

    plot.save()
    
    # Assignment 1: e-greedy
    
    # Assignment 2: Optimistic init
    
    # Assignment 3: UCB
    
    # Assignment 4: Comparison

if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31
    assignment = 2
    
    experiment(n_actions=n_actions,n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window,assignment=assignment)