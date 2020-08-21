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
#thisWorld.set_timestep_player_penalty(-10)
#thisWorld.set_border_collide_penalty(-500)
thisWorld.set_goal_captured_reward(5000)
thisWorld.set_player_timelimit_penalty(-5000)

# Learning Params
thisWorld.set_global_learning_rate(0.15)
thisWorld.set_global_discount_factor(0.25)

# Set Policies
#thisWorld.add_agent_temporary_policy("Player_1", "random", 25000)
#thisWorld.add_agent_temporary_policy("Player_1", "e-greedy", 25000, 0.5)
thisWorld.set_agent_default_policy("Player_1", "e-greedy", 0)

# Run Pygame Simulation
thisWorld.train_agents(300)
thisWorld.run_game()

#thisWorld.plot_agent_network_error("Player_1")