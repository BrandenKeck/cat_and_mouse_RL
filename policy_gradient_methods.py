# Import external libraries
import numpy as np

# Primary Class used to construct PG Methods
class policy_gradients():

    def __init__(self, na):

        # Policy Gradient Learning Settings
        self.num_actions = na
        self.episode_queue_length = 100
        self.num_training_episodes = 10
        self.episode_training_delay = 5

        # Set Policy Gradient Method
        self.use_normally_distributed_policy = True

        # Initialize Policy Objects
        self.policy_parameters = None
        self.state_action_network = None
        self.current_policy = np.ones(na)/na

        # Initialize Episode Objects
        self.active_episode = None
        self.episodes = []


    def REINFORCE(self, next_state, next_state_is_terminal, next_reward, last_action):

        if self.policy_parameters == None: self.initialize_policy_parameterization(next_state)

    def initialize_policy_parameterization(self, next_state):

        if self.use_normally_distributed_policy:
            pass


    # Function for updating rolling queues
    def update_queue(self, queue, obj, length):
        queue.append(obj)
        if length > 0:
            while (len(queue) > length): queue.pop(0)


# Episode Class for Organization of Data
class episode():

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.returns = []