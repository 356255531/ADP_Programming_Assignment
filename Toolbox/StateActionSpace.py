import numpy as np
import sys
import itertools
from copy import deepcopy

from Enviroment import Enviroment


class StateActionSpace(object):
    """
        A class record the legal and illegal state space,
        action space and the rule of transition between state
        and state feature vector

        Member functions:
            __derive_legal_state_space(, env):

            get_state_space()

            get_legal_state_space()

            get_action_space()

            get_feature_vector_of_legal_state(state):

        Attributes:
            __env:
                object intercopy of Enviroment class

            __state_space:
                all states intercopy(all states), list of tuple

            __legal_state_space:
                legal states intercopy, list of tuple

            __action_space:
                intercopy of action space, list of string
    """

    def __init__(self, env):
        super(StateActionSpace, self).__init__()
        self.__env = env
        self.__state_space = env.get_state_space()
        self.__legal_state_space = self.__derive_legal_state_space(env)
        self.__action_space = env.get_action_space()

    def __derive_legal_state_space(self, env):
        """
            derive the legal states from state space

            return a list of tuple
        """
        legal_state_space = env.get_state_space()
        legal_state_space_copy = env.get_state_space()
        for state in legal_state_space_copy:
            if not env.if_state_legal(state):
                legal_state_space.remove(state)
        legal_state_space_copy = deepcopy(legal_state_space)
        return legal_state_space_copy

    def get_state_space(self):
        """
            return state space(all states), list of tuple
        """
        state_space = deepcopy(self.__state_space)
        return state_space

    def get_legal_state_space(self):
        """
            return legal state space, list of tuple
        """
        legal_state_space = deepcopy(self.__legal_state_space)
        return legal_state_space

    def get_action_space(self):
        """
            return action space, key is action(string),
            value is coordinate difference according to action

            return dict
        """
        action_space = deepcopy(self.__action_space)
        return action_space

    def get_feature_vector_of_legal_state(self, state):
        """
            transform a state(tuple) to a sparse vector

            return (0,0,0...,1,0,...0), numpy.matrix
        """
        self.__env.if_state_out_of_range(state)
        try:
            if not self.__env.if_state_legal(state):
                raise ValueError
        except ValueError:
            sys.exit("Try to get feature vector of illegal state(wall)")

        num_legal_state = len(self.__legal_state_space)
        feature_vector_legal_state = np.zeros(num_legal_state)
        state_index_num = self.__legal_state_space.index(state)
        feature_vector_legal_state[state_index_num] = 1
        return feature_vector_legal_state


if __name__ == "__main__":
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

    state_action_space = StateActionSpace(env)

    print state_action_space.get_legal_state_space()
