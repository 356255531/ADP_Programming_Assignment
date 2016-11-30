import numpy as np
import itertools

from Toolbox import Enviroment
from Toolbox import Reward1, Reward2
from Toolbox import PolicyIterationAsyn, ValueIterationAsyn
# from Toolbox import plot_policy, plot_error

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

epsilon = 10e-5

algorithm_sets = [ValueIterationAsyn, PolicyIterationAsyn]

alpha_sets = [0.1, 0.5, 0.9]

reward_sets = [Reward1, Reward2]

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
                reward,
                alpha,
                epsilon
            )
            algorithm.run()
            algorithm_compare_sets.append(algorithm)

policy_sets = []
for algorithm_instance in algorithm_compare_sets:
    val_func = algorithm_instance.get_val_func()
    policy = algorithm_instance.get_policy()
    policy_sets.append(policy)

# plot_policy(policy_sets, plot_labels)
print algorithm_compare_sets[9].print_val_func()
print algorithm_compare_sets[10].print_val_func()
print algorithm_compare_sets[11].print_val_func()
