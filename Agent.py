# Kenneth Zhang
# CSC580
# Final Project

import random
import numpy as np
import DinoEnv as Dino_env

class Agent:
    def __init__(self, env, params):
        self.env = env
        self.action_space = env.action_space # 3 actions from dino game
        self.state_space = env.state_space # 12 feature for dino game
        self.gamma = params['gamma']
        self.alpha = params['alpha']
        self.epsilon = params['epsilon'] 
        self.epsilon_min = params['epsilon_min'] 
        self.epsilon_decay = params['epsilon_decay']
        self.Q = {}
    
    @staticmethod
    def state_to_int(state_list):
        ' Converts the state into bitwise integer '
        return int("".join(str(x) for x in state_list), 2)
    
    @staticmethod
    def state_to_str(state_list):
        ' Converts the state into a string of bitwise integers '
        return "".join(str(x) for x in state_list)

    def init_state(self, state):
        ' Defines a state into the numpy array '
        current_state = self.state_to_int(state)
        if current_state not in self.Q:
            self.Q[current_state] = np.zeros(self.action_space)

    def select_action(self, state):
        ' Selects action based off epsilon '
        if random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return self.select_greedy(state)
        
    def select_greedy(self, state):
        ' Selects the most valued action '
        current_state = self.state_to_int(state)
        if current_state not in self.Q:
            self.init_state(state)
            return np.random.choice(self.action_space)
        return np.argmax(self.Q[current_state])
    
    def adjust_epsilon(self):
        ' Epsilon decay over steps '
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_Qtable(self, state, action, reward, next_state):
        ' Updates the Q value table '
        current_state = self.state_to_int(state)
        next = self.state_to_int(next_state)

        if current_state not in self.Q:
            self.init_state(state)
        if next not in self.Q:
            self.init_state(next_state)
        
        ' Select best action off the greedy method '
        best = self.select_greedy(next_state)
        self.Q[current_state][action] =  self.Q[current_state][action] + self.alpha * (reward  + self.gamma * (best - self.Q[current_state][action]))
       
        # update the epsilon
        self.adjust_epsilon()

    def write_qtable(self, filepath):
        ' Write the content of the Q-table to an output file '
        with open(filepath, "w") as f:
            f.write("state, action, q_value\n")
            for state in sorted(self.Q.keys(), key=lambda x: int(x)): 
                for action, q_value in sorted(enumerate(self.Q[state]), key=lambda x: x[0]): 
                    f.write(f"{state}, {action}, {q_value}\n")

    def read_qtable(self, filepath):
        ' Read in the Q table saved in a csv file. '
        self.Q = {}
        with open(filepath, 'r') as f:
            next(f)
            for line in f:
                state, action, qValue = line.strip().split(", ")
                state, action = int(state), int(action)
                qValue = float(qValue)

                if state not in self.Q:
                    self.Q[state] = np.zeros(self.action_space)
                
                self.Q[state][action] = qValue