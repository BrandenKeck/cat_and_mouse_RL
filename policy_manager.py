# Import external libraries
import numpy as np

class policy_manager():

    def __init__(self):

        # Store a Default Policy:
        self.default_policy = 3
        self.default_epsilon = 0.025

        # Parameter Lists for Each Policy in the Set
        self.policy_type = []
        self.num_cycles = []
        self.epsilons = []

        # Init counter
        self.cycle_counter = 0

    '''
    POLICY FUNCTIONS
    
    (Type - Policy)
    1 - Random
    2 - e-Greedy
    3 - Normalized Q Table
    
    '''

    def generate_policy(self, action_values):

        if self.policy_type == []:
            if self.default_policy == 1:
                return random_policy(action_values)
            elif self.default_policy == 2:
                return e_greedy_policy(action_values, self.default_epsilon)
            elif self.default_policy == 3:
                return normalized_q_table_soft_policy(action_values)
            else:
                return normalized_q_table_soft_policy(action_values)
        elif self.policy_type[len(self.policy_type)-1] == 1:
            return random_policy(action_values)
        elif self.policy_type[len(self.policy_type)-1] == 2:
            return e_greedy_policy(action_values, self.epsilons[len(self.epsilons)-1])
        elif self.policy_type[len(self.policy_type) - 1] == 3:
            return normalized_q_table_soft_policy(action_values)
        else:
            return normalized_q_table_soft_policy(action_values)

    def update_policies(self):
        if self.num_cycles != []:
            self.cycle_counter = self.cycle_counter + 1
            if self.cycle_counter > self.num_cycles[len(self.num_cycles)-1]:
                self.policy_type.pop(0)
                self.num_cycles.pop(0)
                self.epsilons.pop(0)
                self.cycle_counter = 0

    def add_policy(self, type, num_cycles, e=0.025):
        self.policy_type.append(type)
        self.num_cycles.append(num_cycles)
        self.epsilons.append(e)

def random_policy(action_values):
    n = len(action_values)
    policy = np.zeros(n)
    policy[np.random.randint(n)] = 1
    return policy

def e_greedy_policy(action_values, e):

    n = len(action_values)
    policy = np.zeros(n)

    if np.random.rand() < e:
        policy[np.random.randint(n)] = 1
    else:
        policy[np.array(action_values).tolist().index(max(action_values))] = 1

    return policy

# Heuristic policy for exploration when using a value-table learning method
def normalized_q_table_soft_policy(action_values):

    # Initialize an equally-distributed policy
    policy = np.ones(len(action_values))
    policy = policy/len(action_values)

    # Calculate Normalized Q
    normalized_Q = np.zeros(len(action_values))
    for a in np.arange(len(action_values)):
        normalized_Q[a] = action_values[a] - min(action_values)

    # Create an E-soft-ish policy
    normalized_sum = sum(normalized_Q)
    if normalized_sum == 0:
        return policy
    else:
        for a in np.arange(len(action_values)):
            policy[a] = normalized_Q[a] / normalized_sum

    # return the policy
    return policy

'''
TODO - ADD MORE POLICIES
'''