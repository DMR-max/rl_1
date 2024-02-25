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
from tqdm import tqdm
 
def run_repetitions(assignment, epsilon, n_timesteps, l, pi, env, reward_list):
    for i in range(n_timesteps):
        if assignment == 1:
            a = pi.select_action(epsilon[l]) # select action
        if assignment == 2:
            a = pi.select_action() # select action
        if assignment == 3:
            a = pi.select_action(epsilon[l], i+1)
        r = env.act(a) # sample reward
        pi.update(a,r) # update policy
        reward_list[i] += r # take total reward from all timesteps over all repetitions
    return reward_list

def experiment(n_actions, n_timesteps, n_repetitions, smoothing_window, assignment):
    
    #To Do: Write all your experiment code here
    average_reward_list = []
    assignment_4 = False
    if assignment == 1:
        plot = LearningCurvePlot("plot Epsilon Greedy") # For da plot
        epsilon = [0.01,0.05,0.1,0.25]
    elif assignment == 2:
        plot = LearningCurvePlot("plot Optimistic Initialization") # For da plot
        epsilon = [0.1,0.5,1.0,2.0]
    elif assignment == 3:
        plot = LearningCurvePlot("plot Upper Confidence Bounds") # For da plot
        epsilon = [0.01,0.05,0.1,0.25,0.5,1.0]
    elif assignment == 4:
        plot = LearningCurvePlot("plot best greedy, OI and UCB") # For da plot
        epsilon = [0.05,1.0,0.25]
        assignment_4 = True

    reward_means = np.zeros(len(epsilon))
    for l in tqdm(range(len(epsilon))):
        if assignment_4 == True:
            if l == 0:
                assignment = 1
            elif l == 1:
                assignment = 2
            elif l == 2:
                assignment = 3
        reward_list = np.zeros(n_timesteps)
        optimal_action_counter = np.zeros(n_timesteps)
        for j in range(n_repetitions):
            env = BanditEnvironment(n_actions=n_actions) # Initialize environment
            if assignment == 1:
                pi = EgreedyPolicy(n_actions=n_actions) # Initialize policy
            elif assignment == 2:
                pi = OIPolicy(n_actions=n_actions, initial_value=epsilon[l], learning_rate=0.1)
            elif assignment == 3:
                pi = UCBPolicy(n_actions=n_actions)
           
            reward_list = run_repetitions(assignment, epsilon, n_timesteps, l, pi, env, reward_list)
 
        average_reward_list = reward_list / n_repetitions
        reward_means[l] = sum(reward_list) / (n_repetitions * n_timesteps)

        vector_reward = np.array(average_reward_list) # for curve
        vector_reward_smoothed = smooth(vector_reward, smoothing_window)

        if assignment_4 == True:
            if l == 0:
                plot.add_curve(vector_reward_smoothed, label="Epsilon Greedy")
            elif l == 1:
                plot.add_curve(vector_reward_smoothed, label="Optimistic Initialization")
            elif l == 2:
                plot.add_curve(vector_reward_smoothed, label="UCB")
        elif assignment == 1:
            plot.add_curve(vector_reward_smoothed, label=epsilon[l])
        elif assignment == 2:
            plot.add_curve(vector_reward_smoothed, label=epsilon[l])
        elif assignment == 3:
            plot.add_curve(vector_reward_smoothed, label=epsilon[l])

    plot.save()
    return reward_means

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
    assignment = 1
    
    reward_epsilon = experiment(n_actions=n_actions,n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window, assignment=1)
    print(reward_epsilon)
    reward_2 = experiment(n_actions=n_actions,n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window, assignment=2)
    print(reward_2)
    reward_3 = experiment(n_actions=n_actions,n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window, assignment=3)
    print(reward_3)
    plot = ComparisonPlot("Comparison plot") # For da plot
    plot.add_curve(np.array([0.01,0.05,0.1,0.25]), reward_epsilon, label="Epsilon Greedy")
    plot.add_curve(np.array([0.1,0.5,1.0,2.0]), reward_2, label="Optimistic Initialization")
    plot.add_curve(np.array([0.01,0.05,0.1,0.25,0.5,1.0]), reward_3, label="UCB")
    plot.save()
    
    experiment(n_actions=n_actions,n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window, assignment=4)


