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
 
# run repetitions 
def run_repetitions(assignment, epsilon, n_timesteps, l, pi, env, reward_list):
    for i in range(n_timesteps):
        # for assignment 1 epsilon greedy
        if assignment == 1:
            # select action 
            a = pi.select_action(epsilon[l]) 
        # for assignment 2 optimistic initialization
        if assignment == 2:
            # select action 
            a = pi.select_action()
        # for assignment 3 UCB 
        if assignment == 3: 
            # select action
            a = pi.select_action(epsilon[l], i+1) 
        # sample reward
        r = env.act(a) 
        # update policy
        pi.update(a,r)
        # calculate total reward from current timestep (i)
        reward_list[i] += r 
    return reward_list

def experiment(n_actions, n_timesteps, n_repetitions, smoothing_window, assignment):
    
    # To make the average reward plot
    average_reward_list = []
    # To help make exceptions for assignment 4
    assignment_4 = False
    # plot assignment 1 epsilon greedy
    if assignment == 1:
        plot = LearningCurvePlot("Plot average 500 runs, Epsilon Greedy")
        epsilon = [0.01,0.05,0.1,0.25]
    # plot assignment 2 optimistic initialization
    elif assignment == 2:
        plot = LearningCurvePlot("Plot average 500 runs, Optimistic Initialization") 
        epsilon = [0.1,0.5,1.0,2.0]
    # plot assignment 3 UCB
    elif assignment == 3:
        plot = LearningCurvePlot("Plot average 500 runs, Upper Confidence Bounds")
        epsilon = [0.01,0.05,0.1,0.25,0.5,1.0]
    # plot assignment 4 comparison between epsilon greedy, optimistic initialization and UCB
    elif assignment == 4:
        plot = LearningCurvePlot("Plot average 500 runs, best greedy, OI and UCB")
        epsilon = [0.05,1.0,0.25]
        assignment_4 = True

    # make a list of zeros for recording the average reward
    reward_means = np.zeros(len(epsilon))
    # for loop for all epsilon values
    for l in range(len(epsilon)):
        # for loop to go through all the assignments for the graph for assignment 4
        if assignment_4 == True:
            if l == 0:
                assignment = 1
            elif l == 1:
                assignment = 2
            elif l == 2:
                assignment = 3
        # list of all rewards in the current repetition
        reward_list = np.zeros(n_timesteps)
        # for loop which does the repetitions
        for j in range(n_repetitions):
            # initalize environment and policy depending on the assignment
            env = BanditEnvironment(n_actions=n_actions) 
            if assignment == 1:
                pi = EgreedyPolicy(n_actions=n_actions) 
            elif assignment == 2:
                pi = OIPolicy(n_actions=n_actions, initial_value=epsilon[l], learning_rate=0.1)
            elif assignment == 3:
                pi = UCBPolicy(n_actions=n_actions)

            # get reward list from the repetitions function
            reward_list = run_repetitions(assignment, epsilon, n_timesteps, l, pi, env, reward_list)
        # calculate the average reward for the current epsilon / inital value / C for assignment 1,2,3
        average_reward_list = reward_list / n_repetitions
        # calculate the average reward for the current epsilon / inital value / C for assignment 4a
        reward_means[l] = sum(reward_list) / (n_repetitions * n_timesteps)
        # smooth the average reward list
        vector_reward = np.array(average_reward_list) 
        vector_reward_smoothed = smooth(vector_reward, smoothing_window)
        # add the correct curve to the corresponding plot with the correct labels
        if assignment_4 == True:
            if l == 0:
                plot.add_curve(vector_reward_smoothed, label="Epsilon Greedy")
            elif l == 1:
                plot.add_curve(vector_reward_smoothed, label="Optimistic Initialization")
            elif l == 2:
                plot.add_curve(vector_reward_smoothed, label="UCB")
        elif assignment == 1:
            plot.add_curve(vector_reward_smoothed, label= "ε = " + str(epsilon[l]))
        elif assignment == 2:
            plot.add_curve(vector_reward_smoothed, label="initial value = " +str(epsilon[l]))
        elif assignment == 3:
            plot.add_curve(vector_reward_smoothed, label="C = " + str(epsilon[l]))
    # save plot with the correct title
    if assignment_4:
        plot.save("Optimal Greedy, OI and UCB")
    elif assignment == 1:
         plot.save("Epsilon Greedy")
    elif assignment == 2:
        plot.save("Optimistic Initialization")
    elif assignment == 3:
        plot.save("UCB")
    # return the reward means for the comparison plot (4a)
    return reward_means

    

if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31
    assignment = 1
    
    # run experiment for assignment 1
    reward_epsilon = experiment(n_actions=n_actions,n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window, assignment=1)
    # run experiment for assignment 2
    reward_2 = experiment(n_actions=n_actions,n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window, assignment=2)
    # run experiment for assignment 3
    reward_3 = experiment(n_actions=n_actions,n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window, assignment=3)
    # run experiment for assignment (4b)
    experiment(n_actions=n_actions,n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window, assignment=4)
    # make the comparison plot (4a)
    plot = ComparisonPlot("Comparison plot")
    plot.add_curve(np.array([0.01,0.05,0.1,0.25]), reward_epsilon, label="Epsilon Greedy")
    plot.add_curve(np.array([0.1,0.5,1.0,2.0]), reward_2, label="Optimistic Initialization")
    plot.add_curve(np.array([0.01,0.05,0.1,0.25,0.5,1.0]), reward_3, label="UCB")
    plot.save("Comparison plot")
    
