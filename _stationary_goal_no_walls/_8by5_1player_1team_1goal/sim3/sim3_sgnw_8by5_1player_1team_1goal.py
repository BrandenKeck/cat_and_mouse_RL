# Change system directory
import sys
sys.path.append("../../..")

# Import Game
from gridworld import world

# Create game instance
thisWorld = world(8, 5)
thisWorld.add_player("Player_1", 1, 2)
thisWorld.add_goal("Goal_1", 6, 2)

# Establish settings
thisWorld.set_timestep_player_penalty(-1)
thisWorld.set_border_collide_penalty(-5)
thisWorld.set_goal_captured_reward(50)
thisWorld.set_timelimit(-1)
#thisWorld.set_player_timelimit_penalty(-10000)

# Learning Params
thisWorld.set_global_learning_rate(0.01)
thisWorld.set_global_discount_factor(0.95)

# Set Policies
'''
thisWorld.add_agent_temporary_policy("Player_1", "random", 20)
thisWorld.add_agent_temporary_policy("Player_1", "e-greedy", 20, 0.75)
thisWorld.add_agent_temporary_policy("Player_1", "e-greedy", 20, 0.5)
thisWorld.add_agent_temporary_policy("Player_1", "e-greedy", 20, 0.25)
thisWorld.add_agent_temporary_policy("Player_1", "e-greedy", 20, 0.1)'''
#thisWorld.set_agent_default_policy("Player_1", "e-greedy", 0.05)

# Run Pygame Simulation
thisWorld.train_agents(1000)
thisWorld.run_game()

thisWorld.plot_agent_network_error("Player_1")