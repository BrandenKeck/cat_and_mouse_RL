# Get libraries
import pickle
import numpy as np

# Get custom classes
from policy_manager import policy_manager
from qtable import qtable
from qnetwork import qnetwork

# Define a class to be used universally by all players, including movable goals
class player():

    def __init__(self, name, img, x, y):

        # Initialize player attributes
        self.name = name
        self.img = img
        self.team = 1

        # Initialize position for episode memory
        self.init_x = x
        self.init_y = y

        # Position attributes
        self.x = x
        self.y = y
        self.target_x = x
        self.target_y = y
        self.pos_x = 25 * x
        self.pos_y = 25 * y
        self.target_pos_x = 25 * x
        self.target_pos_y = 25 * y

        # Learning params and attributes
        self.alpha = 0.1
        self.gamma = 0.65
        self.last_reward = 0
        self.last_action = 0
        self.current_policy = []

        # Attempt to get stored learning structures
        self.qtable, self.qnetwork = self.get_player_data()
        self.policy_manager = policy_manager()
        self.save_after_iter = 5000
        self.save_after_iter_counter = self.save_after_iter

        # Learning Settings for the agent
        self.use_q_learning = False
        self.use_dqn = True

    '''
    BEGIN PRIMARY FUNCTIONS
    '''

    # Utilize learning classes based on user settings
    # Update current policy as a result
    def learn(self, current_state):

        # Handle First Pass for Learning Objects
        if self.qtable == None: self.qtable = qtable(5, self.alpha, self.gamma)
        if self.qnetwork == None: self.qnetwork = qnetwork(5, [50, 20, 5], self.alpha, self.gamma)

        # Q-Learning Method uses the QTable Custom Class Object
        if self.use_q_learning:

            # Learn the QTable
            self.qtable.q_learning(current_state, self.last_reward, self.last_action)
            action_values = self.qtable.action_values[self.qtable.curr_state_idx].tolist()

        # DQN Method uses the QNetwork Custom Class Object
        if self.use_dqn:

            # Learn using DQN method
            self.qnetwork.dqn(np.array(current_state).flatten().tolist(), self.last_reward, self.last_action)
            action_values = self.qnetwork.action_values

        # Update Policy based on user settings
        self.current_policy = self.policy_manager.generate_policy(action_values)

        # Write to player file to save learned states, policies, and Q functions
        self.save_after_iter_counter = self.save_after_iter_counter + 1
        if self.save_after_iter_counter >= self.save_after_iter:
            self.set_player_data()
            self.save_after_iter_counter = 0


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
        if action == 1: self.set_target(0, -1)
        elif action == 2: self.set_target(1, 0)
        elif action == 3: self.set_target(0, 1)
        elif action == 4: self.set_target(-1, 0)

        # Store previous action
        self.last_action = action

    '''
    BEGIN MISC FUNCTIONS
    '''

    # Search for stored state/policy file
    def get_player_data(self):
        try:
            # Read stored player information if it exists
            with open(self.name + ".txt", 'rb') as file:
                qtable, qnetwork = pickle.load(file)
            return qtable, qnetwork
        except:
            # Initialize empty player information if no file found
            return None, None

    # Store player data for use in subsequent simulations
    def set_player_data(self):
        try:
            # Write Player Data to Stored File
            with open(self.name + ".txt", 'wb') as file:
                pickle.dump((self.qtable, self.qnetwork), file)
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