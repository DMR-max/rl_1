#banditExperiment:
#line 30 ===========================================
#To Do: Write all your experiment code here
    # plot = LearningCurvePlot("plot Epsilon Greedy") # For da plot
    # epsilon = [0.01]
    # for l in range(len(epsilon)):
    #     average_reward_list = []
    #     for j in tqdm(range(n_repetitions)):
    #         reward_list = []
    #         env = BanditEnvironment(n_actions=n_actions) # Initialize environment 
    #         pi = EgreedyPolicy(n_actions=n_actions) # Initialize policy
    #         # print("Mean pay-off per action: {}".format(env.means))
    #         # print("Best action = {} with mean pay-off {}".format(env.best_action,env.best_average_return))

    #         for i in range(n_timesteps):
    #             a = pi.select_action(epsilon[l]) # select action
    #             r = env.act(a) # sample reward
    #             pi.update(a,r) # update policy
    #             reward_list.append(r)
    #             # print("Test e-greedy policy with action {}, received reward {}".format(a,r))
    #         average_this_rep = sum(reward_list) / n_repetitions
    #         average_reward_list.append(average_this_rep)

    #         # print(pi.action_val) # printing final values this repetition
    #     vector_reward = np.array(average_reward_list, dtype=object) # for curve
    #     vector_reward_smoothed = smooth(vector_reward, smoothing_window)
    #     plot.add_curve(vector_reward_smoothed)
    # plot.save()


    # plot = LearningCurvePlot("plot Epsilon Greedy") # For da plot
    # epsilon = [0.01,0.05,0.1,0.25]
    # for l in range(len(epsilon)):
    #     total_rewards = np.zeros(n_timesteps)
    #     for j in tqdm(range(n_repetitions)):
    #         env = BanditEnvironment(n_actions=n_actions) # Initialize environment 
    #         pi = EgreedyPolicy(n_actions=n_actions) # Initialize policy
    #         for i in range(n_timesteps):
    #             a = pi.select_action(epsilon[l]) # select action
    #             r = env.act(a) # sample reward
    #             pi.update(a,r) # update policy
    #             total_rewards[i] += r
    #     average_rewards = total_rewards / n_repetitions
    #     average_rewards_smoothed = smooth(average_rewards, smoothing_window)
    #     plot.add_curve(average_rewards_smoothed, epsilon[l])
    # plot.save()

 # print("Mean pay-off per action: {}".format(env.means))
            # print("Best action = {} with mean pay-off {}".format(env.best_action,env.best_average_return))

#stuk 2 bij line 69 ===========================================
               # print("best_action: ", env.best_action)
                # print("a: ", a)
                # optimal_action_counter[i] += int(a == env.best_action)
                # print("optimal_action_counter_updated: ", optimal_action_counter)
                # print("Test e-greedy policy with action {}, received reward {}".format(a,r))
            
        # if assignment == 2:
        #     # print("optimal_action_counter: ", optimal_action_counter)
        #     # print("n_timesteps: ", n_timesteps)
        #     average_reward_list = optimal_action_counter / n_repetitions # so you know what on average the reward per timestep is
        # else:

#line 72 ===========================================
 # print(pi.action_val) # printing final values this repetition



#BanditPolicies:
# line 26 ===========================================
 # a = np.random.randint(0, self.n_actions) # Replace this with correct action selection
#line 38 ===========================================
        # return a
        #np choice
        # for a in range(self.n_actions):
        #     if self.action_val[a] == np.argmax(self.action_val):
        #         self.action_mean[a] = 1 - epsilon
        #         print(self.action_mean[a])
        #     else:
        #         self.action_mean[a] = (epsilon / (self.n_actions - 1)) / (self.n_actions - 1)
        #         print("else")
        #         print(self.action_mean[a])
        # choice = np.random.choice(10, p=self.action_mean)

        # if choice < 10*epsilon:
        #     a = np.argmax(self.action_val)
        # else:
        #     a = np.random.randint(0, self.action_numb)

        # best_action = np.argmax(self.action_val)
        # # print(best_action)
        # probabilities = np.full(self.n_actions - 1, epsilon / (self.n_actions - 1.0))
        # # print(probabilities)
        # # probabilities[best_action] += (1.0 - epsilon)
        # probabilities = np.insert(probabilities, best_action, (1.0 - epsilon))
        # # print(probabilities)
        # # print(probabilities)
        # a = np.random.choice(np.arange(self.n_actions), p=probabilities)
        # print(a)
#line 43 ===========================================
# value = ((r-self.action_val[a]) / (self.action_tried[a]))
#line 47 ===========================================
        # if self.action_tried[a] == 500:
        #     print(self.action_val)
#line 57 ===========================================
        # TO DO: Add own code
        #a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        # print("est_list: ", self.est_action_val)
#line 62 ===========================================
# print("r: ", r)
# line 64 ===========================================
        # print("r-old: ", value)
        # print("old_value: ", self.est_action_val[a])
#line 66 ===========================================
        # print("new_value: ", self.est_action_val[a])
        # print('est list')
        # print(self.est_action_val)
#line 88 ===========================================
# value = ((r-self.action_val[a]) / (self.action_tried[a]))
#line 91 ===========================================
                # print(value2)
        # print(value)