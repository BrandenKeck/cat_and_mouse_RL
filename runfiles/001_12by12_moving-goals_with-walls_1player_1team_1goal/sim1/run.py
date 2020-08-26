# Change system directory
import sys
sys.path.append("../../..")

# Import Game
from cat_and_mouse import world

# Create game instance
thisWorld = world(12, 12)
thisWorld.add_player("Player_1", 1, 1)
thisWorld.add_goal("Goal_1", 10, 10)
for i in range(12):
    for j in range(12):
        if (((i == 5 or i == 6) and (j > 1 and j < 10)) or ((j == 5 or j == 6) and (i > 1 and i < 10))):
            thisWorld.add_wall(i, j)

# Establish Player settings
thisWorld.set_timestep_player_penalty(-10)
thisWorld.set_border_collide_penalty(-5)
thisWorld.set_wall_collide_penalty(-5)
thisWorld.set_goal_captured_reward(100)
#thisWorld.set_player_goal_reward_gradient_factor(1000)
thisWorld.set_timelimit(-1)

# Establish goal settings
thisWorld.set_goal_caught_penalty(-1000)
thisWorld.set_timestep_goal_reward(1)
thisWorld.set_movable_goals(True)

# Learning Params
thisWorld.set_global_learning_rate(0.001)
thisWorld.set_global_discount_factor(0.95)


# Set Policies
'''
thisWorld.add_agent_temporary_policy("Player_1", "random", 20)
thisWorld.add_agent_temporary_policy("Player_1", "e-greedy", 20, 0.75)
thisWorld.add_agent_temporary_policy("Player_1", "e-greedy", 20, 0.5)
thisWorld.add_agent_temporary_policy("Player_1", "e-greedy", 20, 0.25)
thisWorld.add_agent_temporary_policy("Player_1", "e-greedy", 20, 0.1)
'''
# Run Pygame Simulation
#thisWorld.train_agents(10000)
thisWorld.run_game()

thisWorld.plot_agent_network_error("Player_1")