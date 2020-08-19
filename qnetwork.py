# Library Imports
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

# Custom Imports
from neural_network import neural_network

# Create a Q-Learning Setup for Deep Q Learning
class qnetwork():

    def __init__(self, num_actions, hidden_layer_sizes, learning_rate, discount_factor):

        # Init number of possible actions per state (network output layer)
        # Init hidden layer sizes
        self.num_actions = num_actions
        self.hidden_layer_sizes = hidden_layer_sizes

        # Init Learning Parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Init Empty Learning objects
        self.replay_memory = replay_memory()
        self.Q = []
        self.Q_target = []
        self.action_values = np.ones(num_actions)

        # DQN Parameters
        self.replay_memory_capacity = 800
        self.network_reset_frequency = 500
        self.network_minibatch_size = 100
        self.network_training_delay = 2
        self.network_training_iter = 10
        self.state_queue_length = 4

        # Init counters
        self.network_training_delay_counter = 0
        self.network_reset_counter = 0

    # Run the DQN method
    def dqn(self, state, prev_reward, prev_action):

        # Initialize DQN Neural Network Objects if first pass
        if self.Q == [] or self.Q_target == []:
            self.initialize_q_network(state)
            self.Q_target = deepcopy(self.Q)
            return

        # Update Standard Replay Memory Queues
        self.update_replay_memory(state, prev_reward, prev_action)
        if len(self.replay_memory.preprocessed_states) < 2: return

        # Get target labels from Q network predictions
        predicted_Q = []
        for q in self.Q_target:
            predicted_Q.append(q.classify_data(self.replay_memory.preprocessed_states[len(self.replay_memory.preprocessed_states) - 1]))

        target_label = (prev_reward + self.discount_factor * max(predicted_Q))
        self.update_queue(self.replay_memory.q_target_labels, target_label.tolist()[0][0], self.replay_memory_capacity)

        # Train the Q networks
        self.train_q_networks()

        # Update target Q network and labels
        self.network_reset_counter = self.network_reset_counter + 1
        if self.network_reset_counter >= self.network_reset_frequency:
            self.update_dqn_target()
            self.network_reset_counter = 0

        # Set current action values from the action-value network
        self.action_values = []
        for q in self.Q:
            self.action_values.append(q.classify_data(self.replay_memory.preprocessed_states[len(self.replay_memory.preprocessed_states) - 1])[0][0])


    # Training function to be enacted on a collection of neural networks used to predict the value of each possible action
    def train_q_networks(self):

        # If the training delay has been met, train the action-value network
        self.network_training_delay_counter = self.network_training_delay_counter + 1
        if self.network_training_delay_counter > self.network_training_delay:
            for iter in np.arange(self.network_training_iter):
                for i in np.arange(self.num_actions):
                    action_idx = [j for j, x in enumerate(self.replay_memory.actions) if x == i]
                    minibatch_size = min(len(action_idx), self.network_minibatch_size)
                    minibatch_selections = np.random.choice(len(action_idx), size=minibatch_size, replace=False)
                    minibatch_idx = np.array(action_idx)[minibatch_selections]
                    if len(minibatch_idx) < 1: return

                    X = np.concatenate(self.replay_memory.preprocessed_states, axis=1)[:, minibatch_idx]
                    Y = np.array(self.replay_memory.q_target_labels).reshape(1, -1)[:, minibatch_idx]

                    self.Q[i].predict(X)
                    self.Q[i].learn(Y)

                    self.network_training_delay_counter = 0

    # Function to update the Q target network and associated target labels
    def update_dqn_target(self):

        # If a reset step has been reached, set the target network equal to the current action-value network
        self.Q_target = deepcopy(self.Q)

        # Update target labels for the new network
        X = np.concatenate(self.replay_memory.preprocessed_states, axis=1)
        pred_values = []
        for q in self.Q_target:
            pred_values.append(q.classify_data(X))

        pred_values = np.concatenate(pred_values, axis=1).reshape(5, -1)
        max_pred_values = np.amax(pred_values, axis=0)[1:]
        rewards = np.array(self.replay_memory.rewards)
        labels = rewards + self.discount_factor * max_pred_values
        self.replay_memory.q_target_labels = labels.tolist()

    # Function to update universally-used state, action, and reward queues
    def update_replay_memory(self, state, prev_reward, prev_action):

        # Create a queue of last states to be concatenated for the Q learning network input
        # If the state queue has not been initialized to the required length, return to parent function
        self.update_queue(self.replay_memory.state_queue, state, self.state_queue_length)
        if len(self.replay_memory.state_queue) < self.state_queue_length: return

        # Concatenate state queue to create network input
        # If the previous concatenated state is blank return to parent function
        # Leave one additional pre-processed state in the max queue due to prev-state-next-state paradigm
        preprocessed_state = [state for s in self.replay_memory.state_queue for state in s]
        preprocessed_state = np.array(preprocessed_state).reshape((len(preprocessed_state), 1))
        self.update_queue(self.replay_memory.preprocessed_states, preprocessed_state, self.replay_memory_capacity+1)
        if len(self.replay_memory.preprocessed_states) < 2: return

        # Update action and reward to the appropriate queues
        self.update_queue(self.replay_memory.actions, prev_action, self.replay_memory_capacity)
        self.update_queue(self.replay_memory.rewards, prev_reward, self.replay_memory_capacity)

    # Create Q using external Neural Network class
    def initialize_q_network(self, state):

        # Create a list of layer sizes from network settings
        layer_sizes = [self.state_queue_length * np.size(state)]
        for hl in self.hidden_layer_sizes:
            layer_sizes.append(hl)
        layer_sizes.append(1)

        # Create the Q neural networks (one for each action) and init basic settings
        for i in np.arange(self.num_actions):
            self.Q.append(neural_network(layer_sizes))
            self.Q[i].learning_rates = self.learning_rate * np.ones(len(layer_sizes))
            self.Q[i].use_sigmoid[len(self.Q[i].use_sigmoid) - 1] = False
            self.Q[i].use_linear[len(self.Q[i].use_linear) - 1] = True

    # Function for updating rolling queues
    def update_queue(self, queue, obj, length):
        queue.append(obj)
        if length > 0:
            while(len(queue) > length): queue.pop(0)

    def plot_network_errors(self):
        aes = ['k-', 'r-', 'b-', 'g-', 'm-', 'k--', 'r--', 'b--', 'g--', 'm--', 'k-.', 'r-.', 'b-.', 'g-.', 'm-.', 'k.', 'r.', 'b.', 'g.', 'm.']
        for i, q in enumerate(self.Q):
            plt.plot(np.arange(len(q.squared_errors)), q.squared_errors, aes[i])

        plt.show()

# Create a separate class structure for replay memory lists
class replay_memory():

    def __init__(self):
        self.state_queue = []
        self.preprocessed_states = []
        self.actions = []
        self.rewards = []
        self.q_target_labels = []