import numpy as np
import itertools

from Toolbox import Enviroment
from Toolbox import Reward1, Reward2
from Toolbox import ValueIterationSyn, PolicyIterationSyn
from Toolbox import StateActionSpace
from Toolbox import PlotAgent

__auther__ = "Zhiwei"

##################### Enviroment Setting #####################
maze_map = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ],
    bool
)

index_range = [i for i in xrange(0, 8)]
state_space = [i for i in itertools.product(index_range, repeat=2)]

action_space = ["up", "down", "left", "right"]

action_dict = {
    "up": [-1, 0],
    "down": [1, 0],
    "left": [0, -1],
    "right": [0, 1]
}

start_state = (4, 5)
goal_state = (6, 6)

env = Enviroment(maze_map,
                 state_space,
                 action_space,
                 action_dict,
                 start_state,
                 goal_state
                 )

################### Dependent Module Setting ################
state_action_space = StateActionSpace(env)

########################## RL parameter setting ###############
epsilon = 10e-5

alpha = 0.9

################ Comparison Condition Setting ################
algorithm_sets = [ValueIterationSyn, PolicyIterationSyn]

alpha_sets = [0.1, 0.2, 0.6, 0.9]

reward_sets = [Reward1, Reward2]

###################### Comparing ############################
algorithm_compare_sets = []
plot_labels = []
for algorithm_setting in algorithm_sets:
    for reward_setting in reward_sets:
        for alpha in alpha_sets:
            reward = reward_setting(env)
            plot_labels.append(algorithm_setting.__name__ +
                               ', ' +
                               reward_setting.__name__ +
                               ', ' +
                               'alpha = ' +
                               str(alpha))
            algorithm = algorithm_setting(
                env,
                state_action_space,
                reward,
                alpha,
                epsilon
            )
            algorithm.run()
            if (
                algorithm_setting == ValueIterationSyn
            ) and (reward_setting == Reward1) and (alpha == 0.6):
                optimal_value_function_reward1 = algorithm.get_val_func_vector()
            if (
                algorithm_setting == ValueIterationSyn
            ) and (reward_setting == Reward2) and (alpha == 0.6):
                optimal_value_function_reward2 = algorithm.get_val_func_vector()

            algorithm_compare_sets.append(algorithm)

###################### Record Policy #########################
policy_sets = []
error_sets = []
val_func_vector_sets = []
for algorithm_instance in algorithm_compare_sets:
    error = algorithm_instance.get_error()
    error_sets.append(error)
    policy = algorithm_instance.get_policy()
    policy_sets.append(policy)
    val_func_vector = algorithm_instance.get_val_func_vector()
    val_func_vector_sets.append(val_func_vector)

###################### Plot and save images ####################
plot_agent = PlotAgent(env, state_action_space)  # note that all the images will be saved
plot_agent.plot_policies(policy_sets, plot_labels)  # but not displayed
plot_agent.plot_error(error_sets, plot_labels)
plot_agent.plot_val_func(
    val_func_vector_sets,
    plot_labels,
    state_action_space,
    env
)

algorithm_compare_sets = []
plot_labels = []
for algorithm_setting in algorithm_sets:
    for reward_setting in reward_sets:
        for alpha in alpha_sets:
            reward = reward_setting(env)
            plot_labels.append(algorithm_setting.__name__ +
                               ', ' +
                               reward_setting.__name__ +
                               ', ' +
                               'alpha = ' +
                               str(alpha))
            if (reward_setting == Reward1):
                algorithm = algorithm_setting(
                    env,
                    state_action_space,
                    reward,
                    alpha,
                    epsilon,
                    optimal_value_function_reward1
                )
            else:
                algorithm = algorithm_setting(
                    env,
                    state_action_space,
                    reward,
                    alpha,
                    epsilon,
                    optimal_value_function_reward2
                )
            algorithm.run()
            algorithm_compare_sets.append(algorithm)

###################### Record Policy #########################
policy_sets = []
error_sets = []
val_func_vector_sets = []
for algorithm_instance in algorithm_compare_sets:
    error = algorithm_instance.get_error()
    error_sets.append(error)

###################### Plot and save images ####################
plot_agent = PlotAgent(env, state_action_space)  # note that all the images will be saved
plot_agent.plot_error(error_sets, plot_labels, 'ErrorToGroundTruth.jpg')
