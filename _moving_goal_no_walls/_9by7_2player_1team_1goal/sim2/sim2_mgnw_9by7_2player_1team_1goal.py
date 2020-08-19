# Change system directory
import sys
sys.path.append("../../..")

# Import Game
from gridworld import world

# Create game instance
thisWorld = world(9, 7)
thisWorld.add_player("Player_1", 1, 3)
thisWorld.add_player("Player_2", 7, 3)
thisWorld.add_goal("Goal_1", 5, 3)

# Establish settings
thisWorld.set_border_collide_penalty(-50)
thisWorld.set_teammate_collide_penalty(-50)
thisWorld.set_global_learning_rate(0.05)
thisWorld.set_global_discount_factor(0.75)
thisWorld.set_timelimit_penalty(-10000)

# Establish settings - Player
thisWorld.set_timestep_player_penalty(-1000)
#thisWorld.set_goal_gradient_reward_factor(100)
thisWorld.set_goal_reward(5000)
thisWorld.set_team_goal_reward(3000)

# Establish settings - Goal
thisWorld.set_timestep_goal_reward(1)
#thisWorld.set_goal_player_repulsion_factor(100)
thisWorld.set_goal_caught_penalty(-10000)

# Run Pygame Simulation
thisWorld.set_movable_goals(True)
#thisWorld.train_agents(20000)
thisWorld.run_game()