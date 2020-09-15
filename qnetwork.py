# Library Imports
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

# Custom Imports
from neural_network import neural_network

# Create a Q-Learning Setup for Deep Q Learning
class qnetwork():

    def __init__(self, na, hls, lr, df, rmc, nrf, nms, ntd, nti, sql):

        # Init number of possible actions per state (network output layer)
        # Init hidden layer sizes
        self.num_actions = na
        self.hidden_layer_sizes = hls

        # Init Learning Parameters
        self.learning_rate = lr
        self.discount_factor = df

        # Init Empty Learning objects
        self.replay_memory = replay_memory()
        self.Q = []
        self.Q_target = []
        self.action_values = np.ones(na)

        # DQN Parameters
        self.training_mode = True
        self.replay_memory_capacity = rmc
        self.network_reset_frequency = nrf
        self.network_minibatch_size = nms
        self.network_training_delay = ntd
        self.network_training_iter = nti
        self.state_queue_length = sql

        # Init counters
        self.network_training_delay_counter = 0
        self.network_reset_counter = 0

    # Run the DDQN method
    def ddqn(self, next_state, next_state_is_terminal, next_reward, prev_action):
        self.dqn(next_state, next_state_is_terminal, next_reward, prev_action, True)

    # Run the DQN method
    def dqn(self, next_state, next_state_is_terminal, next_reward, prev_action, double_dqn=False):

        # Initialize DQN Neural Network Objects if first pass
        if self.Q == [] or self.Q_target == []:
            self.initialize_q_network(next_state)
            self.Q_target = deepcopy(self.Q)
            return

        # Update Standard Replay Memory Queues
        self.update_replay_memory(next_state, next_state_is_terminal, next_reward, prev_action)
        if next_state == None or len(self.replay_memory.preprocessed_states) < 6: return

        # Set current action values from the action-value network
        self.action_values = []
        for q in self.Q:
            self.action_values.append(q.classify_data(
                self.replay_memory.preprocessed_states[len(self.replay_memory.preprocessed_states) - 1])[0][0])

        # If training is not enabled, return to parent function
        if not self.training_mode:
            return

        # Train the Q networks (training delay configurable)
        self.network_training_delay_counter = self.network_training_delay_counter + 1
        if self.network_training_delay_counter > self.network_training_delay:
            self.train_q_networks(double_dqn)
            self.network_training_delay_counter = 0

        # If a reset step has been reached, set the target network equal to the current action-value network
        self.network_reset_counter = self.network_reset_counter + 1
        if self.network_reset_counter >= self.network_reset_frequency:
            self.Q_target = deepcopy(self.Q)
            self.network_reset_counter = 0

    '''
    TRAINING FUNCTIONS
    '''

    # Training function to be enacted on a collection of neural networks used to predict the value of each possible action
    def train_q_networks(self, double_dqn):

        # Create the Target Data from the Target Network
        if double_dqn:
            self.update_ddqn_target()
        else:
            self.update_dqn_target()

        # Trim Datasets For Training (2 values excluded from front and back of array)
        # Only one value is trimmed from the from of the state array so that previous states are aligned with target labels
        prev_state_set = self.replay_memory.preprocessed_states[1:len(self.replay_memory.preprocessed_states) - 3]
        action_set = self.replay_memory.actions[2:len(self.replay_memory.preprocessed_states) - 2]

        # Train each of the action-value networks
        for iter in np.arange(self.network_training_iter):
            for i in np.arange(self.num_actions):
                action_idx = [j for j, x in enumerate(action_set) if x == i]
                minibatch_size = min(len(action_idx), self.network_minibatch_size)
                minibatch_selections = np.random.choice(len(action_idx), size=minibatch_size, replace=False)
                minibatch_idx = np.array(action_idx)[minibatch_selections]
                if len(minibatch_idx) < 1: return

                X = np.concatenate(prev_state_set, axis=1)[:, minibatch_idx]
                Y = np.array(self.replay_memory.q_target_labels).reshape(1, -1)[:, minibatch_idx]

                self.Q[i].predict(X)
                self.Q[i].learn(Y)

    # Function to update the Q target network and associated target labels
    def update_ddqn_target(self):

        # Trim Datasets For Training (2 values excluded from front and back of array)
        state_set = self.replay_memory.preprocessed_states[2:len(self.replay_memory.preprocessed_states) - 2]
        reward_set = self.replay_memory.rewards[2:len(self.replay_memory.preprocessed_states) - 2]

        # Get Q network predictions
        X = np.concatenate(state_set, axis=1)
        q_predictions = []
        for q in self.Q:
            q_predictions.append(q.classify_data(X))

        # Get action indices of max. predicted values from Q network
        q_predictions = np.concatenate(q_predictions, axis=1).reshape(5, -1)
        max_pred_values_idx = np.argmax(q_predictions, axis=0)[0:]

        # Get Target Network Predictions from Max. Q Network Valued Actions
        q_target_predictions = []
        for idx, a in enumerate(max_pred_values_idx):
            q_target_predictions.append(self.Q_target[a].classify_data(X[:, idx]))

        # Compute the target labels
        q_target_predictions = np.concatenate(q_target_predictions).reshape(1, -1)
        rewards = np.array(reward_set)
        labels = rewards + self.discount_factor * q_target_predictions
        self.replay_memory.q_target_labels = labels.tolist()

    # Function to update the Q target network and associated target labels
    def update_dqn_target(self):

        # Trim Datasets For Training (2 values excluded from front and back of array)
        state_set = self.replay_memory.preprocessed_states[2:len(self.replay_memory.preprocessed_states)-2]
        reward_set = self.replay_memory.rewards[2:len(self.replay_memory.preprocessed_states)-2]

        # Get Q Target Network Predictions
        X = np.concatenate(state_set, axis=1)
        pred_values = []
        for q in self.Q_target:
            pred_values.append(q.classify_data(X))

        # Find the maximum action value for each state
        pred_values = np.concatenate(pred_values, axis=1).reshape(5, -1)
        max_pred_values = np.amax(pred_values, axis=0)[0:]

        # Compute the target labels
        rewards = np.array(reward_set)
        labels = rewards + self.discount_factor * max_pred_values
        self.replay_memory.q_target_labels = labels.tolist()

    # Function to update universally-used state, action, and reward queues
    def update_replay_memory(self, next_state, next_state_is_terminal, next_reward, prev_action):

        # Break on start of new simulation, reuse previous state terminal flag for simulation start instance
        if next_state == None:
            self.replay_memory.previous_state_terminal = True
            return

        # Create a queue of last states to be concatenated for the Q learning network input
        # If the state queue has not been initialized to the required length, return to parent function
        self.update_queue(self.replay_memory.state_queue, next_state, self.state_queue_length)
        if len(self.replay_memory.state_queue) < self.state_queue_length: return

        # Handle Current State Terminal Case
        if next_state_is_terminal:
            self.update_queue(self.replay_memory.actions, None, self.replay_memory_capacity)
            self.update_queue(self.replay_memory.preprocessed_states_is_terminal, True, self.replay_memory_capacity)
            self.replay_memory.previous_state_terminal = True
        # Handle Previous State is terminal case
        elif self.replay_memory.previous_state_terminal:
            self.update_queue(self.replay_memory.actions, prev_action, self.replay_memory_capacity)
            self.replay_memory.previous_state_terminal = False
            self.replay_memory.twice_previous_state_terminal = True
            return
        # Handle Twice Previous State is terminal case
        elif self.replay_memory.twice_previous_state_terminal:
            self.update_queue(self.replay_memory.preprocessed_states_is_terminal, False, self.replay_memory_capacity)
            self.replay_memory.twice_previous_state_terminal = False
        # Handle Normal State Case
        else:
            self.update_queue(self.replay_memory.actions, prev_action, self.replay_memory_capacity)
            self.update_queue(self.replay_memory.preprocessed_states_is_terminal, False, self.replay_memory_capacity)

        # Concatenate state queue to create network input
        # Update the state queue and reward queue appropriately
        preprocessed_state = [state for s in self.replay_memory.state_queue for state in s]
        preprocessed_state = np.array(preprocessed_state).reshape((len(preprocessed_state), 1))
        self.update_queue(self.replay_memory.preprocessed_states, preprocessed_state, self.replay_memory_capacity+1)
        self.update_queue(self.replay_memory.rewards, next_reward, self.replay_memory_capacity)

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
            self.Q[i].learning_rates = self.learning_rate * np.ones((len(layer_sizes) - 1))
            self.leaky_relu_rates = 0.01 * np.ones(len(layer_sizes) - 1)
            self.huber_cost_delta = 5
            self.Q[i].use_huber_cost = True
            self.Q[i].use_hellinger_cost = False
            self.Q[i].use_quadratic_cost = False
            self.Q[i].use_cross_entropy_cost = False
            self.Q[i].use_merged_softmax_cross_entropy_cost = False
            self.Q[i].use_leaky_relu = [True] * (len(layer_sizes) - 2) + [True]
            self.Q[i].use_softmax = [False] * (len(layer_sizes) - 1)
            self.Q[i].use_sigmoid = [False] * (len(layer_sizes) - 2) + [False]
            self.Q[i].use_relu = [False] * (len(layer_sizes) - 1)
            self.Q[i].use_linear = [False] * (len(layer_sizes) - 2) + [False]
            self.Q[i].use_tanh = [False] * (len(layer_sizes) - 1)

    # Function for updating rolling queues
    def update_queue(self, queue, obj, length):
        queue.append(obj)
        if length > 0:
            while(len(queue) > length): queue.pop(0)

    # Function to plot mean squared errors of the Q networks
    def plot_network_errors(self):
        aes = ['k-', 'r-', 'b-', 'g-', 'm-', 'k--', 'r--', 'b--', 'g--', 'm--', 'k-.', 'r-.', 'b-.', 'g-.', 'm-.', 'k.', 'r.', 'b.', 'g.', 'm.']
        for i, q in enumerate(self.Q):
            plt.plot(np.arange(len(q.diag.mse)), q.diag.mse, aes[i])

        plt.show()

# Create a separate class structure for replay memory lists
class replay_memory():

    def __init__(self):
        self.previous_state_terminal = False
        self.twice_previous_state_terminal = False
        self.state_queue = []
        self.preprocessed_states = []
        self.preprocessed_states_is_terminal = []
        self.actions = []
        self.rewards = []
        self.q_target_labels = []