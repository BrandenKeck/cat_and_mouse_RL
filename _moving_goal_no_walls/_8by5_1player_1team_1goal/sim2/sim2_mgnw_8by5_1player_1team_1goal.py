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
thisWorld.set_border_collide_penalty(-50)
thisWorld.set_global_learning_rate(0.01)
thisWorld.set_global_discount_factor(0.2)

# Establish settings - Player
thisWorld.set_timestep_player_penalty(-100)
#thisWorld.set_goal_gradient_reward_factor(100)
thisWorld.set_goal_reward(2000)

# Establish settings - Goal
thisWorld.set_timestep_goal_reward(1)
#thisWorld.set_goal_player_repulsion_factor(100)
thisWorld.set_goal_caught_penalty(-500)

# Run Pygame Simulation
thisWorld.set_movable_goals(True)
#thisWorld.train_agents(1000)
thisWorld.run_game()