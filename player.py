# Get libraries
import pickle
import numpy as np

# Get custom classes
from policy_manager import policy_manager
from qtable import qtable
from qnetwork import qnetwork
from policy_gradient_methods import policy_gradients

# Define a class to be used universally by all players, including movable goals
class player():

    def __init__(self, name, img, x, y, num_actions):

        # Initialize player attributes
        self.name = name
        self.team = 1
        self.num_actions = num_actions
        self.img = img

        # Initialize position for episode memory
        self.init_x = x
        self.init_y = y

        # Position attributes (game params)
        self.x = x
        self.y = y
        self.target_x = x
        self.target_y = y
        self.pos_x = 25 * x
        self.pos_y = 25 * y
        self.target_pos_x = 25 * x
        self.target_pos_y = 25 * y

        # Learning params and attributes
        self.alpha = 0.01
        self.gamma = 0.85
        self.last_action = 0
        self.current_reward = 0
        self.possible_next_states = []
        self.possible_current_states = []
        self.current_state = None
        self.current_state_is_terminal = False
        self.current_policy = np.ones(num_actions)/num_actions

        # Learning object structure / storage parameters
        self.reset_stored_training_data = False
        self.qtable, self.qnetwork, self.policy_gradients = self.get_player_data()
        self.qnetwork_hidden_layer_sizes = [256, 128]
        self.qnetwork_replay_memory_capacity = 1000
        self.qnetwork_network_reset_frequency = 250
        self.qnetwork_network_minibatch_size = 100
        self.qnetwork_network_training_delay = 50
        self.qnetwork_network_training_iter = 10
        self.qnetwork_state_queue_length = 1
        self.policy_manager = policy_manager()
        self.save_after_iter = 5000
        self.save_after_iter_counter = self.save_after_iter

        # Learning Settings for the agent
        self.use_q_learning = True
        self.use_dqn = False
        self.use_ddqn = False
        self.use_REINFORCE = False

    '''-----------------
    BEGIN PRIMARY FUNCTIONS
    -----------------'''

    # Take Action Based on Current Policy
    def act(self):

        # Create a cumulative policy distribution
        pol_sum = 0
        cumprobs = []
        for p in self.current_policy:
            pol_sum = pol_sum + p
            cumprobs.append(pol_sum)

        # Randomly chose an action based on policy
        action = 0
        diceroll = np.random.random(1)[0]
        for idx, c in enumerate(cumprobs):
            if diceroll <= c:
                action = idx
                break

        # Take the decided action
        # action 0 does nothing
        if action == 1:
            self.set_target(0, -1)
        elif action == 2:
            self.set_target(1, 0)
        elif action == 3:
            self.set_target(0, 1)
        elif action == 4:
            self.set_target(-1, 0)

        # Store previous action
        self.last_action = action

    # Utilize learning classes based on user settings
    # Update current policy as a result
    def learn(self):

        # Handle First Pass for Learning Objects
        if self.qtable == None: self.qtable = qtable(self.num_actions, self.alpha, self.gamma)
        if self.qnetwork == None: self.qnetwork = qnetwork(self.num_actions, self.qnetwork_hidden_layer_sizes, self.alpha, self.gamma,
                                                           self.qnetwork_replay_memory_capacity,
                                                           self.qnetwork_network_reset_frequency,
                                                           self.qnetwork_network_minibatch_size,
                                                           self.qnetwork_network_training_delay,
                                                           self.qnetwork_network_training_iter,
                                                           self.qnetwork_state_queue_length)
        if self.policy_gradients == None: self.policy_gradients = policy_gradients(self.num_actions)
        '''
        TODO
        Pass params as PG inputs
        '''

        # Initialize Uniform Action Values / Error Prevention / No Learning Method Selected
        action_values = np.ones(self.num_actions)

        # Q-Learning Method uses the QTable Custom Class Object
        if self.use_q_learning:

            # Learn the QTable
            self.qtable.q_learning(self.current_state, self.current_state_is_terminal, self.current_reward, self.last_action)
            action_values = self.qtable.action_values[self.qtable.next_state_idx].tolist()

        # DQN Method uses the QNetwork Custom Class Object
        elif self.use_dqn:

            # Learn using DQN method
            self.qnetwork.dqn(np.array(self.current_state).flatten().tolist(), self.current_state_is_terminal, self.current_reward, self.last_action)
            action_values = self.qnetwork.action_values

        # DDQN Method uses the QNetwork Custom Class Object
        elif self.use_ddqn:

            # Learn using DDQN method
            self.qnetwork.ddqn(np.array(self.current_state).flatten().tolist(), self.current_state_is_terminal, self.current_reward, self.last_action)
            action_values = self.qnetwork.action_values

        # REINFORCE Method uses the Policy Gradients Custom Class Object
        elif self.use_REINFORCE:

            # Learn using the REINFORCE method
            self.policy_gradients.use_REINFORCE = True
            self.policy_gradients.learn_policy_gradient(self.possible_next_states, self.possible_current_states, np.array(self.current_state).flatten().tolist(), self.current_state_is_terminal, self.current_reward, self.last_action)
            action_values = self.policy_gradients.current_policy

        # Update Policy based on user settings
        self.current_policy = self.policy_manager.generate_policy(action_values)

        # Write to player file to save learned states, policies, and Q functions
        self.save_after_iter_counter = self.save_after_iter_counter + 1
        if self.save_after_iter_counter >= self.save_after_iter:
            self.set_player_data()
            self.save_after_iter_counter = 0

    '''-----------------
    BEGIN MISC FUNCTIONS
    -----------------'''

    # Search for stored state/policy file
    def get_player_data(self):
        if not self.reset_stored_training_data:
            try:
                # Read stored player information if it exists
                with open(self.name + ".txt", 'rb') as file:
                    qtable, qnetwork, policy_gradients = pickle.load(file)
                policy_gradients.episodes = []
                return qtable, qnetwork, policy_gradients
            except:
                # Initialize empty player information if no file found
                return None, None, None
        else:
            return None, None

    # Store player data for use in subsequent simulations
    def set_player_data(self):
        try:
            # Write Player Data to Stored File
            with open(self.name + ".txt", 'wb') as file:
                pickle.dump((self.qtable, self.qnetwork, self.policy_gradients), file)
        except:
            pass

    # Quick move function for use in non-visualized training
    def quick_move(self):
        self.x = self.target_x
        self.y = self.target_y
        self.pos_x = self.target_pos_x
        self.pos_y = self.target_pos_y

    # Purely asthetic function for animating the move of each agent
    def animate_move(self):
        if (self.pos_x != self.target_pos_x or self.pos_y != self.target_pos_y):
            self.pos_x = self.pos_x + np.sign(self.target_pos_x - self.pos_x)
            self.pos_y = self.pos_y + np.sign(self.target_pos_y - self.pos_y)
        else:
            self.x = self.target_x
            self.y = self.target_y

    # Set new position based on agent actions
    def set_target(self, dx, dy):
        self.target_x = self.x + 1 * dx
        self.target_y = self.y + 1 * dy
        self.target_pos_x = self.pos_x + 25 * dx
        self.target_pos_y = self.pos_y + 25 * dy