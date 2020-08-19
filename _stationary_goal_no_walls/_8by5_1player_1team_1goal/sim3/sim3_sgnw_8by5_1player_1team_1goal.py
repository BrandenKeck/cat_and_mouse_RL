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
thisWorld.set_timestep_player_penalty(-1000)
thisWorld.set_border_collide_penalty(-500)
thisWorld.set_goal_captured_reward(5000)

# Learning Params
thisWorld.set_global_learning_rate(0.005)
thisWorld.set_global_discount_factor(0.95)

# Set Policies
thisWorld.add_agent_temporary_policy("Player_1", "random", 25000)
#thisWorld.add_agent_temporary_policy("Player_1", "e-greedy", 100, 0.1)

# Run Pygame Simulation
thisWorld.train_agents(30000)
#thisWorld.run_game()

thisWorld.plot_agent_network_error("Player_1")