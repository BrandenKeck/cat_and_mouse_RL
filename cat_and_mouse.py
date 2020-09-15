# Standard Imports
import sys, os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
from copy import deepcopy

# Custom Class Imports
import player, wall, spritesheet

# World Class - Runs Simulations
class world():

    def __init__(self, xsize, ysize):

        # Dimensions of the board
        self.xsize = xsize
        self.ysize = ysize
        self.w = 25*xsize
        self.h = 25*ysize

        # Set episodic variables
        self.timelimit = 1000
        self.episode_count = 0
        self.current_time = 0
        self.take_action = True

        # Penalties and Rewards
        self.goal_captured_reward = 0
        self.player_goal_reward_gradient_factor = 0
        self.goal_caught_penalty = 0
        self.goal_player_repulsion_factor = 0
        self.team_goal_captured_reward = 0
        self.opponent_goal_captured_penalty = 0
        self.border_collide_penalty = 0
        self.wall_collide_penalty = 0
        self.teammate_collide_penalty = 0
        self.opponent_collide_penalty = 0
        self.goal_agent_collide_penalty = 0
        self.timestep_player_penalty = 0
        self.timestep_goal_reward = 0
        self.player_timelimit_penalty = 0
        self.goal_timelimit_reward = 0

        # Movable Goals Parameters
        self.movable_goals = False

        # Initialize Arrays
        self.players = []
        self.goals = []
        self.walls = []
        self.goal_rewards_gradient = []
        self.goal_repulsion_gradient = []

        # Image Properties
        self.player_imgs = None
        self.goal_imgs = None
        self.wall_imgs = None

    # Pygame Function to Display Visual Simulations
    def run_game(self):

        # Output info to command line:
        print("")
        print("RUNNING GAME")
        print("")

        # Initialize game
        pygame.init()
        window = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("gridworld")
        self.set_images()
        run = True

        # Start the game loop
        while run:

            # Refresh window
            pygame.time.delay(10)
            pygame.draw.rect(window, (0, 0, 0), (0, 0, self.w, self.h))

            # Calculate Gradient Terms
            self.goal_rewards_gradient = self.get_goal_rewards_gradient()
            self.goal_repulsion_gradient = self.get_goal_repulsion_gradient()

            # Perform learning functions and enact policy
            self.simulate_action(False)

            # Draw objects
            self.draw(window)

            # Update Display
            pygame.display.update()

            # Exit on Esc
            for event in pygame.event.get():
                if (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    run = False

        # End the Game
        pygame.quit()

    # Training Function to run simulations without display
    def train_agents(self, num_episodes):

        # Output info to command line:
        print("")
        print("TRAINING AGENTS")
        print("Required Episode Completions:  " + str(num_episodes))
        print("")

        self.episode_count = 0
        while self.episode_count < num_episodes:

            # Calculate Gradient Terms
            self.goal_rewards_gradient = self.get_goal_rewards_gradient()
            self.goal_repulsion_gradient = self.get_goal_repulsion_gradient()

            # Simulate
            self.simulate_action(True)

        # Force Save Agent Data at end of training cycles
        for p in self.players:
            p.set_player_data()
        if self.movable_goals:
            for g in self.goals:
                g.set_player_data()

    # Simulation Runner
    # Calculates all player/goal rewards and updates positions
    def simulate_action(self, quickact):

        # Next State Achieved, Apply Learning Functions, Enable Next Action
        if self.update_ready() and (not self.take_action):

            # Check if Last Action Completed the Episode
            # Handle rewards if episode was completed
            self.current_time = self.current_time + 1
            self.check_episode_completion()

            # Get all possible next states
            for p in self.players: p.possible_next_states = self.get_next_states(p, 1)
            for g in self.goals: g.possible_next_states = self.get_next_states(g, -1)

            # Apply Learning to Players
            for p in self.players:
                p.current_state = self.get_current_state(p.x, p.y, 1)
                p.learn()

            # Apply Learning to Movable Goals
            if self.movable_goals:
                for g in self.goals:
                    g.current_state = self.get_current_state(g.x, g.y, -1)
                    g.learn()

            # System Ready for next action
            self.take_action = True

        # Check if all moves have completed before taking an action
        if self.take_action:

            # Get all possible current states
            for a in self.players + self.goals: a.possible_current_states = deepcopy(a.possible_next_states)

            # Act and Reward Functions
            self.attempt_agent_actions()
            self.check_env_collisions()
            self.check_player_to_player_collisions()
            self.calculate_rewards_gradients()
            self.calculate_timestep()
            if self.movable_goals: self.check_movable_goal_collisions()

            # Action Complete - Pause for Simulation
            self.take_action = False

        # Execute movement of player
        for p in self.players:
            if quickact: p.quick_move()
            else: p.animate_move()

        # Execute movement of goals
        if self.movable_goals:
            for g in self.goals:
                if quickact: g.quick_move()
                else: g.animate_move()

    # Check that animations have completed for "Run" mode
    def update_ready(self):
        for p in self.players:
            if p.target_x != p.x or p.target_y != p.y: return False
        for g in self.goals:
            if g.target_x != g.x or g.target_y != g.y: return False

        return True

    # If a goal is caught, reset the simulation
    # Assign rewards and penalties accordingly
    def check_episode_completion(self):

        # Handle timelimit reached
        if self.current_time >= self.timelimit and self.timelimit > 0:

            for p in self.players:
                p.current_reward = self.player_timelimit_penalty
                p.current_state_is_terminal = True
            for g in self.goals:
                g.current_reward = self.goal_timelimit_reward
                g.current_state_is_terminal = True

            # Reset simulation and print results
            self.reset()
            self.episode_count = self.episode_count + 1
            print("Completed Episodes:  " + str(self.episode_count) + " (EPISODE TIMEOUT)")

            return True

        # Check all player and goal positions for goal caught
        for p in self.players:
            for g in self.goals:
                if p.x == g.x and p.y == g.y:
                    p.current_reward = self.goal_captured_reward
                    p.current_state_is_terminal = True
                    for pp in self.players:
                        if pp != p and p.team == pp.team:
                            pp.current_reward = self.team_goal_captured_reward
                        elif pp != p and p.team != pp.team:
                            pp.current_reward = self.opponent_goal_captured_penalty
                        pp.current_state_is_terminal = True

                    g.current_reward = self.goal_caught_penalty
                    g.current_state_is_terminal = True

                    # Reset simulation and print results
                    self.reset()
                    self.episode_count = self.episode_count + 1
                    print("Completed Episodes:  " + str(self.episode_count))

                    return True

                g.current_state_is_terminal = False

            p.current_state_is_terminal = False

        return False

    # Runs Action Functions for all Agents
    def attempt_agent_actions(self):
        for p in self.players:
            p.current_reward = 0
            p.act()
        if self.movable_goals:
            for g in self.goals:
                g.current_reward = 0
                g.act()

    # Border and Wall Collision Penalty for All Agents
    def check_env_collisions(self):
        for a in self.players + self.goals:

            # Border Collisions
            if a.target_x < 0 or a.target_x >= self.xsize:
                a.target_x = a.x
                a.target_pos_x = a.pos_x
                a.current_reward = a.current_reward + self.border_collide_penalty
            if a.target_y < 0 or a.target_y >= self.ysize:
                a.target_y = a.y
                a.target_pos_y = a.pos_y
                a.current_reward = a.current_reward + self.border_collide_penalty

            # Wall Collisions
            for w in self.walls:
                if a.target_x == w.x and a.target_y == w.y:
                    a.target_x = a.x
                    a.target_y = a.y
                    a.target_pos_x = a.pos_x
                    a.target_pos_y = a.pos_y
                    a.current_reward = a.current_reward + self.wall_collide_penalty

    # Player - to - Player Collision Check
    def check_player_to_player_collisions(self):
        for p in self.players:
            for pp in self.players:
                if pp != p and (p.target_x == pp.x and p.target_y == pp.y):
                    if p.team == pp.team: p.current_reward = p.current_reward + self.teammate_collide_penalty
                    else: p.current_reward = p.current_reward + self.opponent_collide_penalty
                    p.target_x = p.x
                    p.target_y = p.y
                    p.target_pos_x = p.pos_x
                    p.target_pos_y = p.pos_y

    # Player and goal rewards gradients
    def calculate_rewards_gradients(self):
        for p in self.players:
            p.current_reward = p.current_reward + self.player_goal_reward_gradient_factor * (self.goal_rewards_gradient[p.target_x][p.target_y] - self.goal_rewards_gradient[p.x][p.y])
        for g in self.goals:
            if self.movable_goals:
                g.current_reward = g.current_reward + self.goal_player_repulsion_factor * (self.goal_repulsion_gradient[g.x][g.y] - self.goal_repulsion_gradient[g.target_x][g.target_y])

    # Timestep penalties and rewards
    def calculate_timestep(self):
        for p in self.players: p.current_reward = p.current_reward + self.timestep_player_penalty
        for g in self.goals: g.current_reward = g.current_reward + self.timestep_goal_reward

    # Check any attempt for a goal to move into an occupied space
    def check_movable_goal_collisions(self):
        for g in self.goals:
            for a in self.players + self.goals:
                if g!=a and (g.target_x == a.x and g.target_y == a.y):
                    g.current_reward = g.current_reward + self.goal_agent_collide_penalty

    # Create a list of lists containing all possible next states for a given Player
    def get_next_states(self, a, flip_sign):

        # Initialize Next States List
        next_states = []

        # No Action Taken
        next_states.append(self.get_current_state(a.x, a.y, flip_sign))

        # Initialize Flags to determine if each move is possible
        Directions = 4 * [True]

        # Get a list of all relevant objects
        objects = self.players + self.walls
        if flip_sign == -1: objects = objects + self.goals

        # Attempt Moves
        for o in objects:
            if (a != o and a.x == o.x and (a.y - 1) == o.y) or ((a.y - 1) < 0): Directions[0] = False
            if (a != o and (a.x + 1) == o.x and a.y == o.y) or ((a.x + 1) >= self.xsize): Directions[1] = False
            if (a != o and a.x == o.x and (a.y + 1)) == o.y or ((a.y + 1) >= self.ysize): Directions[2] = False
            if (a != o and (a.x - 1) == o.x and a.y == o.y) or ((a.x - 1) < 0): Directions[3] = False

        # Append North Action
        if Directions[0]: next_states.append(self.get_current_state(a.x, a.y - 1, flip_sign))
        else: next_states.append(self.get_current_state(a.x, a.y, flip_sign))

        # Append East Action
        if Directions[1]: next_states.append(self.get_current_state(a.x + 1, a.y, flip_sign))
        else: next_states.append(self.get_current_state(a.x, a.y, flip_sign))

        # Append South Action
        if Directions[2]: next_states.append(self.get_current_state(a.x, a.y + 1, flip_sign))
        else: next_states.append(self.get_current_state(a.x, a.y, flip_sign))

        # Append West Action
        if Directions[3]: next_states.append(self.get_current_state(a.x - 1, a.y, flip_sign))
        else: next_states.append(self.get_current_state(a.x, a.y, flip_sign))

        return next_states

    # Create 2D list representation of the current state
    def get_current_state(self, xx, yy, flip_sign):
        state = np.zeros([self.xsize, self.ysize])
        for p in self.players:
            state[p.x][p.y] = -10*p.team
        for g in self.goals:
            state[g.x][g.y] = 100
        for w in self.walls:
            state[w.x][w.y] = 10

        state[xx][yy] = flip_sign*-100

        return np.array(state).flatten().tolist()

    # Create rewards gradient based on normalized inverse manhattan distance from each goal
    def get_goal_rewards_gradient(self):
        ness = np.zeros([self.xsize, self.ysize])
        onett = 0
        for i in np.arange(self.xsize):
            for j in np.arange(self.ysize):
                for g in self.goals:
                    if (np.abs(i - g.x) + np.abs(j - g.y)) != 0:
                        ness[i][j] = ness[i][j] + 1/(np.abs(i - g.x) + np.abs(j - g.y))
                        onett = onett + 1/(np.abs(i - g.x) + np.abs(j - g.y))
                    else:
                        ness[i][j] = 1

        for i in np.arange(self.xsize):
            for j in np.arange(self.ysize):
                ness[i][j] = ness[i][j]/onett

        return ness

    # Create goal repulsion gradient based on normalized inverse manhattan distance from each player
    def get_goal_repulsion_gradient(self):
        ness = np.zeros([self.xsize, self.ysize])
        onett = 0
        for i in np.arange(self.xsize):
            for j in np.arange(self.ysize):
                for p in self.players:
                    if (np.abs(i - p.x) + np.abs(j - p.y)) != 0:
                        ness[i][j] = ness[i][j] + 1 / (np.abs(i - p.x) + np.abs(j - p.y))
                        onett = onett + 1 / (np.abs(i - p.x) + np.abs(j - p.y))
                    else:
                        ness[i][j] = 1

        for i in np.arange(self.xsize):
            for j in np.arange(self.ysize):
                ness[i][j] = ness[i][j] / onett

        return ness

    # Start of a new episode - set everything back to original positions
    def reset(self):
        self.current_time = 0
        for a in self.players + self.goals:
            a.x = a.init_x
            a.y = a.init_y
            a.target_x = a.init_x
            a.target_y = a.init_y
            a.pos_x = 25 * a.init_x
            a.pos_y = 25 * a.init_y
            a.target_pos_x = 25 * a.init_x
            a.target_pos_y = 25 * a.init_y
            a.policy_manager.update_policies()

    '''---------------------
    BEGIN RENDERING FUNCTIONS
    ---------------------'''

    # Draw world objects
    def draw(self, window):
        for p in self.players:
            window.blit(p.img, (p.pos_x, p.pos_y))
        for g in self.goals:
            window.blit(g.img, (g.pos_x, g.pos_y))
        for w in self.walls:
            window.blit(w.img, (w.pos_x, w.pos_y))

    # Import image data and set to variables (Cat / Mouse / Wall characters)
    def set_images(self):
        os.chdir(os.path.dirname(sys.argv[0]))
        self.player_imgs = spritesheet.make_sprite_array(spritesheet.spritesheet('../../../img/cats.png'), 5, 25, 25)
        self.goal_imgs = spritesheet.make_sprite_array(spritesheet.spritesheet('../../../img/mouse.png'), 1, 25, 25)
        self.wall_imgs = spritesheet.make_sprite_array(spritesheet.spritesheet('../../../img/wall.png'), 1, 25, 25)

        for p in self.players:
            p.img = self.player_imgs[p.team - 1]
        for g in self.goals:
            g.img = self.goal_imgs[0]
        for w in self.walls:
            w.img = self.wall_imgs[0]

    '''---------------------
    BEGIN SET/GET FUNCTIONS
    ---------------------'''

    ###
    # WORLD CONFIG FUNCTIONS
    ###

    # Adds elements to the world with name 'name'  (except walls)
    # Elements are added at position '(x,y)'
    # Note that the y-axis is inverted per image notation
    def add_player(self, name, x, y): self.players.append(player.player(name, None, x, y, 5))
    def add_goal(self, name, x, y): self.goals.append(player.player(name, None, x, y, 5))
    def add_wall(self, x, y): self.walls.append(wall.wall(None, x, y))

    # Configurable Goal Rewards (/Penalties) Settings
    def set_movable_goals(self, val): self.movable_goals = val
    def set_goal_captured_reward(self, val): self.goal_captured_reward = val
    def set_team_goal_captured_reward(self, val): self.team_goal_captured_reward = val
    def set_opponent_goal_captured_penalty(self, val): self.opponent_goal_captured_penalty = val
    def set_goal_caught_penalty(self, val): self.goal_caught_penalty = val
    def set_player_goal_reward_gradient_factor(self, val): self.player_goal_reward_gradient_factor = val
    def set_goal_player_repulsion_factor(self, val): self.goal_player_repulsion_factor = val

    # Configurable Collision Rewards (/Penalties) Settings
    def set_teammate_collide_penalty(self, val): self.teammate_collide_penalty = val
    def set_opponent_collide_penalty(self, val): self.opponent_collide_penalty = val
    def set_goal_agent_collide_penalty(self, val): self.goal_agent_collide_penalty = val
    def set_border_collide_penalty(self, val): self.border_collide_penalty = val
    def set_wall_collide_penalty(self, val): self.wall_collide_penalty = val

    # Configurable In-Game Timer Rewards (/Penalties) Settings
    def set_timestep_player_penalty(self, val): self.timestep_player_penalty = val
    def set_timestep_goal_reward(self, val): self.timestep_goal_reward = val
    def set_player_timelimit_penalty(self, val): self.player_timelimit_penalty = val
    def set_goal_timelimit_reward(self, val): self.goal_timelimit_reward = val

    # Set Time Limit for each Simulation
    def set_timelimit(self, val):
        self.timelimit = val
        print("TIMELIMIT UPDATED TO " + str(val) + " ITERATIONS")

    ###
    # AGENT LEARNING PARAM FUNCTIONS - APPLY TO BOTH PLAYERS AND GOALS
    ###

    # Functions to set Learning Method (tabular types, network approx types, policy gradient types) for the agents
    def set_global_learning_method(self, method):
        for a in self.players + self.goals:
            if method == "q_learning":
                a.use_q_learning = True
                a.use_dqn = False
                a.use_ddqn = False
            elif method == "dqn":
                a.use_q_learning = False
                a.use_dqn = True
                a.use_ddqn = False
            elif method == "ddqn":
                a.use_q_learning = False
                a.use_dqn = False
                a.use_ddqn = True
            else:
                a.use_q_learning = False
                a.use_dqn = False
                a.use_ddqn = True
    def set_agent_learning_method(self, name, method):
        for a in self.players + self.goals:
            if a.name == name:
                if method == "q_learning":
                    a.use_q_learning = True
                    a.use_dqn = False
                    a.use_ddqn = False
                elif method == "dqn":
                    a.use_q_learning = False
                    a.use_dqn = True
                    a.use_ddqn = False
                elif method == "ddqn":
                    a.use_q_learning = False
                    a.use_dqn = False
                    a.use_ddqn = True
                else:
                    a.use_q_learning = False
                    a.use_dqn = False
                    a.use_ddqn = True
                return
        print("ERROR: PLAYER " + name + " NOT FOUND")

    # Functions for setting the "alpha" (learning rate) parameter, either globally or to an agent by name
    def set_global_learning_rate(self, val):
        for a in self.players + self.goals:
            a.alpha = val
            if a.qtable != None: a.qtable.learning_rate = val
            if a.qnetwork != None:
                a.qnetwork.learning_rate = val
                for q in a.qnetwork.Q:
                    q.learning_rates = val * np.ones(len(a.qnetwork.hidden_layer_sizes) + 2)
    def set_agent_learning_rate(self, name, val):
        for a in self.players + self.goals:
            if a.name == name:
                a.alpha = val
                if a.qtable != None: a.qtable.learning_rate = val
                if a.qnetwork != None:
                    a.qnetwork.learning_rate = val
                    for q in a.qnetwork.Q:
                        q.learning_rates = val * np.ones(len(a.qnetwork.hidden_layer_sizes) + 2)
                return
        print("ERROR: PLAYER " + name + " NOT FOUND")

    # Functions for setting the "gamma" (discount factor) parameter, either globally or to an agent by name
    def set_global_discount_factor(self, val):
        for a in self.players + self.goals:
            a.gamma = val
            if a.qtable != None: a.qtable.discount_factor = val
            if a.qnetwork != None: a.qnetwork.discount_factor = val
    def set_agent_discount_factor(self, name, val):
        for a in self.players + self.goals:
            if a.name == name:
                a.gamma = val
                if a.qtable != None: a.qtable.discount_factor = val
                if a.qnetwork != None: a.qnetwork.discount_factor = val
                return
        print("ERROR: PLAYER " + name + " NOT FOUND")

    # Functions for setting the "epsilon" (non-greedy policy percentage) parameter, either globally or to an agent by name
    def set_global_policy_epsilon(self, val):
        for a in self.players + self.goals:
            a.policy_manager.default_epsilon = val

    def set_agent_policy_epsilon(self, name, val):
        for a in self.players + self.goals:
            if a.name == name:
                a.policy_manager.default_epsilon = val
                return
        print("ERROR: PLAYER " + name + " NOT FOUND")

    ###
    # AGENT POLICY FUNCTIONS - APPLY TO BOTH PLAYERS AND GOALS
    ###

    # Set the default policy for all agents
    def set_global_default_policy(self, policy="normalized-q", epsilon=0.025):
        for a in self.players + self.goals:
            if policy == "random":
                a.policy_manager.default_policy = 1
                print("RANDOM SET AS DEFAULT FOR AGENT " + a.name + ".")
            elif policy == "e-greedy":
                a.policy_manager.default_policy = 2
                print("e-GREEDY (e = " + str(epsilon) + ") SET AS DEFAULT FOR AGENT " + a.name + ".")
            elif policy == "normalized-q":
                a.policy_manager.default_policy = 3
                print("NORMALIZED Q TABLE POLICY SET AS DEFAULT FOR AGENT " + a.name + ".")
            else:
                a.policy_manager.default_policy = 3
                print("NORMALIZED Q TABLE POLICY SET AS DEFAULT FOR AGENT " + a.name + ".")

    # Set the default for an agent by name
    def set_agent_default_policy(self, name, policy="normalized-q", epsilon=0.025):
        for a in self.players + self.goals:
            if a.name == name:
                if policy == "random":
                    a.policy_manager.default_policy = 1
                    print("RANDOM SET AS DEFAULT FOR AGENT " + name + ".")
                elif policy == "e-greedy":
                    a.policy_manager.default_policy = 2
                    print("e-GREEDY (e = " + str(epsilon) + ") SET AS DEFAULT FOR AGENT " + name + ".")
                elif policy == "normalized-q":
                    a.policy_manager.default_policy = 3
                    print("NORMALIZED Q TABLE POLICY SET AS DEFAULT FOR AGENT " + name + ".")
                else:
                    a.policy_manager.default_policy = 3
                    print("NORMALIZED Q TABLE POLICY SET AS DEFAULT FOR AGENT " + name + ".")
                return
        print("ERROR: PLAYER " + name + " NOT FOUND")

    # Configure a new temporary policy for all agents for num_cycles episodes
    def add_global_temporary_policy(self, policy="random", num_cycles=100, epsilon=0.025):
        for a in self.players + self.goals:
            if policy == "random":
                a.policy_manager.add_policy(1, num_cycles)
                print("TEMPORARY RANDOM POLICY ADDED w/ " + str(num_cycles) + " CYCLES FOR ALL AGENTS.")
            elif policy == "e-greedy":
                a.policy_manager.add_policy(2, num_cycles, epsilon)
                print("TEMPORARY e-GREEDY (e = " + str(epsilon) + ") POLICY ADDED w/ " + str(num_cycles) + " CYCLES FOR ALL AGENTS.")
            elif policy == "normalized-q":
                a.policy_manager.add_policy(3, num_cycles)
                print("TEMPORARY NORMALIZED Q TABLE POLICY ADDED w/ " + str(num_cycles) + " CYCLES FOR ALL AGENTS.")
            else:
                a.policy_manager.add_policy(1, num_cycles)
                print("TEMPORARY RANDOM POLICY ADDED w/ " + str(num_cycles) + " CYCLES FOR ALL AGENTS.")

    # Configure a new temporary policy for a specific agent (by name) for num_cycles episodes
    def add_agent_temporary_policy(self, name, policy="random", num_cycles=100, epsilon=0.025):
        for a in self.players + self.goals:
            if a.name == name:
                if policy == "random":
                    a.policy_manager.add_policy(1, num_cycles)
                    print("RANDOM POLICY ADDED w/ " + str(num_cycles) + " CYCLES FOR AGENT " + name + ".")
                elif policy == "e-greedy":
                    a.policy_manager.add_policy(2, num_cycles, epsilon)
                    print("e-GREEDY (e = " + str(epsilon) + ") POLICY ADDED w/ " + str(num_cycles) + " CYCLES FOR AGENT " + name + ".")
                elif policy == "normalized-q":
                    a.policy_manager.add_policy(3, num_cycles)
                    print("NORMALIZED Q TABLE POLICY ADDED w/ " + str(num_cycles) + " CYCLES FOR AGENT " + name + ".")
                else:
                    a.policy_manager.add_policy(1, num_cycles)
                    print("RANDOM POLICY ADDED w/ " + str(num_cycles) + " CYCLES FOR AGENT " + name + ".")
                return
        print("ERROR: PLAYER " + name + " NOT FOUND")

    ###
    # AGENT NETWORK FUNCTIONS - APPLY TO BOTH PLAYERS AND GOALS
    ###

    # Functions for setting the network layersizes
    # AUTOMATICALLY RESETS LEARNING -- DESTROYS EXISTING NETWORKS
    def set_global_qnetwork_hidden_layersizes(self, vals):
        for a in self.players + self.goals:
            a.qnetwork = None
            a.qnetwork_hidden_layer_sizes = vals
    def set_agent_qnetwork_hidden_layersizes(self, name, vals):
        for a in self.players + self.goals:
            if a.name == name:
                a.qnetwork = None
                a.qnetwork_hidden_layer_sizes = vals
                return
        print("ERROR: PLAYER " + name + " NOT FOUND")

    # Functions for setting the capacity (number of stored states/actions/rewards) of an agents' qnetwork replay memory
    def set_global_qnetwork_replay_memory_capacity(self, val):
        for a in self.players + self.goals:
            a.qnetwork_replay_memory_capacity = val
            if a.qnetwork != None: a.qnetwork.replay_memory_capacity = val
    def set_agent_qnetwork_replay_memory_capacity(self, name, val):
        for a in self.players + self.goals:
            if a.name == name:
                a.qnetwork_replay_memory_capacity = val
                if a.qnetwork != None: a.qnetwork.replay_memory_capacity = val
                return
        print("ERROR: PLAYER " + name + " NOT FOUND")

    # Functions for setting the frequency (number of learning steps) before resetting DQN targets
    def set_global_qnetwork_network_reset_frequency(self, val):
        for a in self.players + self.goals:
            a.qnetwork_network_reset_frequency = val
            if a.qnetwork != None: a.qnetwork.network_reset_frequency = val
    def set_agent_qnetwork_network_reset_frequency(self, name, val):
        for a in self.players + self.goals:
            if a.name == name:
                a.qnetwork_network_reset_frequency = val
                if a.qnetwork != None: a.qnetwork.network_reset_frequency = val
                return
        print("ERROR: PLAYER " + name + " NOT FOUND")

    # Functions for setting the number of previous transitions to use for DQN training in each training step
    def set_global_qnetwork_network_minibatch_size(self, val):
        for a in self.players + self.goals:
            a.qnetwork_network_minibatch_size = val
            if a.qnetwork != None: a.qnetwork.network_minibatch_size = val
    def set_agent_qnetwork_network_minibatch_size(self, name, val):
        for a in self.players + self.goals:
            if a.name == name:
                a.qnetwork_network_minibatch_size = val
                if a.qnetwork != None: a.qnetwork.network_minibatch_size = val
                return
        print("ERROR: PLAYER " + name + " NOT FOUND")

    # Functions for setting the number of actions to take before each DQN training step
    def set_global_qnetwork_network_training_delay(self, val):
        for a in self.players + self.goals:
            a.qnetwork_network_training_delay = val
            if a.qnetwork != None: a.qnetwork.network_training_delay = val
    def set_agent_qnetwork_network_training_delay(self, name, val):
        for a in self.players + self.goals:
            if a.name == name:
                a.qnetwork_network_training_delay = val
                if a.qnetwork != None: a.qnetwork.network_training_delay = val
                return
        print("ERROR: PLAYER " + name + " NOT FOUND")

    # Functions for setting the number of training iterations for each DQN training step
    def set_global_qnetwork_network_training_iter(self, val):
        for a in self.players + self.goals:
            a.qnetwork_network_training_iter = val
            if a.qnetwork != None: a.qnetwork.network_training_iter = val
    def set_agent_qnetwork_network_training_iter(self, name, val):
        for a in self.players + self.goals:
            if a.name == name:
                a.qnetwork_network_training_iter = val
                if a.qnetwork != None: a.qnetwork.network_training_iter = val
                return
        print("ERROR: PLAYER " + name + " NOT FOUND")

    # Functions for setting the number of states to concatenate into a single preprocessed state for DQN training
    # AUTOMATICALLY RESETS LEARNING -- DESTROYS EXISTING NETWORKS
    def set_global_qnetwork_state_queue_length(self, val):
        for a in self.players + self.goals:
            a.qnetwork = None
            a.qnetwork_state_queue_length = val
    def set_agent_qnetwork_state_queue_length(self, name, val):
        for a in self.players + self.goals:
            if a.name == name:
                a.qnetwork = None
                a.qnetwork_state_queue_length = val
                return
        print("ERROR: PLAYER " + name + " NOT FOUND")

    ###
    # MISC FUNCTIONS
    ###

    # A function for plotting the squared errors for a particular agent's dqn
    def plot_agent_network_error(self, name):
        for a in self.players + self.goals:
            if a.name == name:
                a.qnetwork.plot_network_errors()

    # Change team of player by player name (Team Default is 1; Options are 1 through 5)
    def set_player_team(self, name, team):
        if team < 1 or team > 5:
            team = 1
            print("ERROR: MAXIMUM NUMBER OF TEAMS IS 5")
            print(">> SETTING PLAYER " + name + " TO TEAM #1")
        for p in self.players:
            if p.name == name:
                p.team = team
                return
        print("ERROR: PLAYER " + name + " NOT FOUND")