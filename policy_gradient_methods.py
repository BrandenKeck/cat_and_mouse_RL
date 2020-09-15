# Import external libraries
import numpy as np
from copy import deepcopy

# Primary Class used to construct PG Methods
class policy_gradients():

    def __init__(self, na):

        # General Learning Settings
        self.num_actions = na
        self.learning_rate = 0.001
        self.discount_factor = 0.85

        # Policy Gradient Learning Settings
        self.state_queue_length = 4
        self.episode_queue_length = 30
        self.num_training_episodes = 15
        self.episode_training_delay = 10
        self.episode_training_delay_counter = 0

        # Set Policy Gradient Method
        self.use_REINFORCE = True

        # Set Policy Type
        self.use_linear_expression_softmax_policy = True

        # Initialize Policy Objects
        self.is_init = False
        self.policy_parameters = None
        self.policy_network = None
        self.state_action_network = None
        self.current_policy = np.ones(na)/na

        # Initialize Episode Objects
        self.episodes = []


    def learn_policy_gradient(self, possible_next_states, possible_current_states, current_state, current_state_is_terminal, current_reward, last_action):

        # Handle empty object case
        if not self.is_init:
            self.policy_parameters = np.random.random(self.state_queue_length * len(current_state))
            self.is_init = True
        if self.episodes == []: self.episodes.append(episode())

        # Update episode information
        if possible_current_states != []: self.update_episode_memory(possible_current_states, current_state, current_state_is_terminal, current_reward, last_action)

        # Get Current Policy - Use Random Policy if next states not provided
        if len(self.episodes) >= self.episode_queue_length and len(self.episodes[len(self.episodes)-1].possible_preprocessed_states) > 0:
            self.get_current_policy(self.episodes[len(self.episodes)-1].possible_preprocessed_states[len(self.episodes[len(self.episodes)-1].possible_preprocessed_states)-1])
        else: self.current_policy = np.ones(self.num_actions)/self.num_actions

        # Train Policy Gradients
        if self.episode_training_delay_counter > self.episode_training_delay:
            self.episode_training_delay_counter = 0
            if self.use_REINFORCE: self.REINFORCE()

    def get_current_policy(self, states):

        if self.use_linear_expression_softmax_policy:
            self.current_policy = self.linear_expression_softmax_policy(states)


    def REINFORCE(self):

        # Train policy parameters
        if len(self.episodes) > self.num_training_episodes + 1:

            # Choose Episodes
            episode_selections = np.random.choice(len(self.episodes)-1, size=self.num_training_episodes, replace=False)

            for e in np.array(self.episodes)[episode_selections]:

                for t in np.arange(len(e.rewards)):

                    returns = 0
                    for tt in np.flip(np.arange(len(e.rewards))):
                        if tt > t: returns = returns + (self.discount_factor**(tt - t - 1)) * e.rewards[tt]
                        else: break

                    if self.use_linear_expression_softmax_policy:
                        policy = self.linear_expression_softmax_policy(e.possible_preprocessed_states[t])
                        expected_f_gradient = np.zeros(self.policy_parameters.shape)
                        for idx, s in enumerate(e.possible_preprocessed_states[t]):
                            expected_f_gradient = expected_f_gradient + policy[idx]*np.array(s).reshape(1, -1)

                        policy_gradient = np.array(e.preprocessed_states[t]).reshape(1, -1) - expected_f_gradient

                    self.policy_parameters = self.policy_parameters - ((self.learning_rate)*(self.discount_factor**t)*(returns)*policy_gradient)[0]


    def linear_expression_softmax_policy(self, states):

        f = []
        for state in states:
            f.append(np.dot(self.policy_parameters, state))

        exps = np.exp(f - np.max(f))
        return (np.array(exps)/np.sum(exps)).reshape(-1,)

    # Function to update episodic state, action, and reward queues
    def update_episode_memory(self, possible_current_states, current_state, current_state_is_terminal, current_reward, prev_action):

        # Break on start of new simulation
        if current_state == None: return

        # Save the previous state queue and update the queue of states to be concatenated into a preprocessed state
        store_episode_state_queue = deepcopy(self.episodes[len(self.episodes) - 1].state_queue)
        self.episodes[len(self.episodes) - 1].state_queue.append(current_state)
        while len(self.episodes[len(self.episodes) - 1].state_queue) > self.state_queue_length: self.episodes[len(self.episodes) - 1].state_queue.pop(0)
        if len(self.episodes[len(self.episodes) - 1].state_queue) < self.state_queue_length: return

        # Update the preprocess state, reward, and action queues for the episode
        preprocessed_state = [state for s in self.episodes[len(self.episodes) - 1].state_queue for state in s]
        preprocessed_state = np.array(preprocessed_state).reshape((len(preprocessed_state), 1))
        self.episodes[len(self.episodes) - 1].preprocessed_states.append(preprocessed_state)
        self.episodes[len(self.episodes) - 1].rewards.append(current_reward)
        if len(self.episodes[len(self.episodes) - 1].rewards) > 1: self.episodes[len(self.episodes) - 2].actions.append(prev_action)

        # Loop through next possible states and create a list of next possible pre-processed states
        possible_preprocessed_states = []
        for ps in possible_current_states:

            # Update the queue of states to be concatenated into a preprocessed state
            state_list = deepcopy(store_episode_state_queue)
            state_list.append(ps)

            while len(state_list) > self.state_queue_length: state_list.pop(0)
            preprocessed_state = [state for s in state_list for state in s]
            preprocessed_state = np.array(preprocessed_state).reshape((len(preprocessed_state), 1))
            possible_preprocessed_states.append(preprocessed_state)

        # Update the possible preprocessed states queue
        self.episodes[len(self.episodes) - 1].possible_preprocessed_states.append(possible_preprocessed_states)

        # If the current state is terminal, append a new episode to the episode queue
        if current_state_is_terminal:
            self.episode_training_delay_counter = self.episode_training_delay_counter + 1
            self.episodes.append(episode())

        # Confirm that the episode queue length remains at the specified size
        while len(self.episodes) > self.episode_queue_length: self.episodes.pop(0)


# Episode Class for Organization of Data
class episode():

    def __init__(self):
        self.state_queue = []
        self.preprocessed_states = []
        self.possible_preprocessed_states = []
        self.actions = []
        self.rewards = []