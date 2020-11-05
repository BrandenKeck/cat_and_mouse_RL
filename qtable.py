# Library Imports
import numpy as np
from copy import deepcopy

# Create a Q-Learning Setup for Tabular Reinforcement Learning
class qtable():

    def __init__(self, num, learning_rate, discount_factor):

        # Init number of possible actions per state
        self.num = num

        # Init Learning Parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.prev_state_terminal = False

        # Init lists
        self.states = []
        self.action_values = []

        # Init counters
        self.prev_state_idx = -1
        self.next_state_idx = -1

    # Define a function for the Q Learning (value-table) method
    def q_learning(self, next_state, next_state_is_terminal, next_reward, last_action):

        # Handle First Pass
        print(next_state)
        if next_state == None: return

        # Attempt to find q table for the current state
        next_values = []
        for idx, s in enumerate(self.states):
            if s == next_state:
                next_values = self.action_values[idx]
                self.next_state_idx = idx

        # Create a new q table and policy for the current state if one doesn't exist
        if next_values == []:
            # Set Default Q Table
            next_values = np.zeros(self.num)

            # Add new state, policy, and q table
            self.states.append(deepcopy(next_state))
            self.action_values.append(np.zeros(self.num))
            self.next_state_idx = len(self.states) - 1

        # Update Q table for prev state, handle first pass, handle terminal state, handle last state terminal:
        if self.prev_state_idx != -1 and (not next_state_is_terminal) and (not self.prev_state_terminal):
            max_next_values = max(next_values)
            self.action_values[self.prev_state_idx][last_action] = self.action_values[self.prev_state_idx][last_action] + self.learning_rate * (next_reward + self.discount_factor * max_next_values - self.action_values[self.prev_state_idx][last_action])
        elif next_state_is_terminal:
            self.action_values[self.prev_state_idx][last_action] = self.action_values[self.prev_state_idx][last_action] + self.learning_rate * (next_reward)
            self.prev_state_terminal = True
        elif self.prev_state_terminal:
            self.prev_state_terminal = False

        # Set Current State to Previous State
        self.prev_state_idx = self.next_state_idx