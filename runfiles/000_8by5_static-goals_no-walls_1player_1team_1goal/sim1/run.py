# Change system directory
import sys
sys.path.append("../../..")

# Import Game
from cat_and_mouse import world

# Create game instance
thisWorld = world(8, 5)
thisWorld.add_player("Player_1", 1, 2)
thisWorld.add_goal("Goal_1", 6, 2)

# Establish Player settings
thisWorld.set_timestep_player_penalty(-10)
thisWorld.set_border_collide_penalty(-5)
thisWorld.set_goal_captured_reward(100)
#thisWorld.set_player_goal_reward_gradient_factor(1000)
thisWorld.set_timelimit(1000)
thisWorld.set_player_timelimit_penalty(-100)

# Establish goal settings
#thisWorld.set_goal_caught_penalty(-1000)
#thisWorld.set_timestep_goal_reward(1)
#thisWorld.set_movable_goals(True)

# Learning Params
thisWorld.set_global_learning_rate(0.0001)
thisWorld.set_global_discount_factor(0.25)

# Q Network Params
thisWorld.set_global_qnetwork_hidden_layersizes([256, 128])
thisWorld.set_global_qnetwork_replay_memory_capacity(500)
thisWorld.set_global_qnetwork_network_reset_frequency(200)
thisWorld.set_global_qnetwork_network_minibatch_size(1)
thisWorld.set_global_qnetwork_network_training_delay(0)
thisWorld.set_global_qnetwork_network_training_iter(5)
thisWorld.set_global_qnetwork_state_queue_length(4)

# Set to random policy for learning
#thisWorld.set_global_default_policy("random")
thisWorld.add_agent_temporary_policy("Player_1", "random", 1000)
thisWorld.add_agent_temporary_policy("Player_1", "e-greedy", 500, 0.5)
thisWorld.add_agent_temporary_policy("Player_1", "e-greedy", 500, 0.25)
thisWorld.add_agent_temporary_policy("Player_1", "e-greedy", 500, 0.1)

# Run Pygame Simulation
thisWorld.train_agents(3000)
thisWorld.run_game()

thisWorld.plot_agent_network_error("Player_1")