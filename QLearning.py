# Kenneth Zhang
# CSC580
# Final Project

import DinoEnv as Dino_env
import Agent as agent_class
import numpy as np
import matplotlib.pyplot as plt
import copy

filepath = '/Users/kennethzhang/Downloads/CSC 580 Final Project/QPath.txt'

def training(agent, env, max_steps, train=True):
    ' Training agent and returns rewards, dones, good steps, and score '
    state = env.reset()
    agent.init_state(state)

    total_rewards, dones, good_steps, max_score = 0.0, 0, 0, 0

    # Runs the training agent for max steps
    for i in range(max_steps):
        if train:
            action = agent.select_action(state)
        else:
            action = agent.select_greedy(state)

        # Output of next state, reward, done, and score for each action takem
        next_state, reward, done, score = env.step(action)

        if train:
            agent.update_Qtable(state, action, reward, next_state)
    
        state = next_state
        total_rewards += pow(agent.gamma, i) * reward

        # Calculate the good steps, dones and max score
        if reward == 20:
            good_steps += 1
        if done:
            dones += 1
        if max_score < score:
            max_score = score

    return total_rewards, dones, good_steps, max_score

def run_train(max_episodes, max_steps, params, qtable_file, train=True):
    ' Runs the training episodes and returns best results'
    episodes = max_episodes
    steps = max_steps
    results = []
    best_return = float('-inf')
    best_qtable = None
    best_score = 0
    best_result = []

    # Runs episodes on the training function
    for ep in range(episodes):
        p = copy.deepcopy(params)

        # Creates dino game
        env = Dino_env.DinoGameEnv(display_game=False)
        # Creates agent
        agent = agent_class.Agent(env,p)

        if not train and qtable_file is not None:
            agent.read_qtable(qtable_file)

        # Collects the output from the training function
        output = training(agent, env, steps,train=train)
        results.append(output)
        env.close()

        # Collects the best returns
        if train:
            if output[0] > best_return:
                best_return = output[0]
                best_dones = output[1]
                best_steps = output[2]
                best_score = output[3]
                best_result = [best_return, best_dones, best_steps, best_score]
                best_qtable = agent.Q
            print(f"Episode: {ep}, Epsilon: {agent.epsilon}")
            print(f'Episode: {ep}, Reward: {best_return}, Dones: {best_dones}, Good Steps: {best_steps}, Score: {best_score}')
        else:
            best_result.append(output[3])

    # Updates q table with best rewards
    if train:
        agent.Q = best_qtable
        agent.write_qtable(filepath)
    return best_result

num_runs = 10
num_steps = 2000

# Parameter and file path set up
qTable = filepath
params = dict()
params['gamma'] = 0.95
params['alpha'] = 0.7
params['epsilon'] = 0.5
params['epsilon_min'] = .01
params['epsilon_decay'] = 0.995

'''
# Training
final_result_train = run_train(num_runs, num_steps, params, qTable, train=True)
x = ['Return', 'Done', 'Best Steps', 'Best Score']
plt.bar(x, final_result_train)
plt.title("Result From Training")
plt.xlabel("Outcomes")
plt.ylabel("Score")
plt.show()
'''

# Testing
final_result_test = run_train(num_runs, num_steps, params, qTable, train=False)

x = range(1,num_runs+1)
plt.plot(x, final_result_test)
plt.title("Score per Run")
plt.xlabel("Run")
plt.ylabel("Score")
plt.show()

''''
# Additional testing

num_runs = 1
num_steps = 10000

final_result_test = run_train(num_runs, num_steps, params, qTable, train=False)

x = range(1,num_runs+1)
plt.plot(x, final_result_test)
plt.title("Score per Run")
plt.xlabel("Run")
plt.ylabel("Score")
plt.show()

# Epsilon testing
params['epsilon'] = .3
results_list = run_train(num_runs, num_steps, params, qTable, train=True)

params['epsilon'] = .5
results_list2 = run_train(num_runs, num_steps, params, qTable, train=True)

params['epsilon'] = .7
results_list3 = run_train(num_runs, num_steps, params, qTable, train=True)

params['epsilon'] = .9
results_list4 = run_train(num_runs, num_steps, params, qTable, train=True)


# Setting up the output for the rewards
epsilon = ['0.3', '0.5', '0.7', '0.9']
rewards = [results_list[0], results_list2[0], results_list3[0], results_list4[0]]
dones = [results_list[1], results_list2[1], results_list3[1], results_list4[1]]
good_steps = [results_list[2], results_list2[2], results_list3[2], results_list4[2]]
scores = [results_list[3], results_list2[3], results_list3[3], results_list4[3]]

plt.plot(epsilon, rewards)
plt.title("Rewards per Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Rewards")
plt.show()

plt.plot(epsilon, dones)
plt.title("Dones per Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Dones")
plt.show()

plt.plot(epsilon, good_steps)
plt.title("Good Steps per Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Good Steps")
plt.show()

plt.plot(epsilon, scores)
plt.title("Max Score per Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Max Score")
plt.show()

# Original testing for epsilon decay
params['epsilon_decay'] = .3
results_list = run_train(num_runs, num_steps, params, qTable, train=True)

params['epsilon_decay'] = .5
results_list2 = run_train(num_runs, num_steps, params, qTable, train=True)

params['epsilon_decay'] = .7
results_list3 = run_train(num_runs, num_steps, params, qTable, train=True)

params['epsilon_decay'] = .9
results_list4 = run_train(num_runs, num_steps, params, qTable, train=True)


# Setting up the output for the rewards
epsilon = ['0.992', '0.995', '0.997', '0.999']
rewards = [results_list[0], results_list2[0], results_list3[0], results_list4[0]]
dones = [results_list[1], results_list2[1], results_list3[1], results_list4[1]]
good_steps = [results_list[2], results_list2[2], results_list3[2], results_list4[2]]
scores = [results_list[3], results_list2[3], results_list3[3], results_list4[3]]

plt.plot(epsilon, rewards)
plt.title("Rewards per Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Rewards")
plt.show()

plt.plot(epsilon, dones)
plt.title("Dones per Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Dones")
plt.show()

plt.plot(epsilon, good_steps)
plt.title("Good Steps per Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Good Steps")
plt.show()

plt.plot(epsilon, scores)
plt.title("Max Score per Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Max Score")
plt.show()

'''