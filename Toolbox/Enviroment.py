import numpy as np
import sys


class Enviroment(object):
    """docstring for Enviroment"""

    def __init__(self,
                 maze,
                 state_space,
                 action_space,
                 action_dict,
                 start_state,
                 goal_state
                 ):
        super(Enviroment, self).__init__()
        self.__maze = maze
        self.__state_space = state_space
        self.__action_space = action_space
        self.__action_dict = action_dict
        self.__start_state = start_state
        self.__goal_state = goal_state

    def get_state_space(self):
        return self.__state_space

    def get_action_space(self):
        return self.__action_space

    def get_action_dict(self):
        return self.__action_dict

    def perform_action(
        self,
        current_state,
        action
    ):
        y, x = list(current_state)

        try:
            if not self.__maze[y, x]:
                raise IndexError
        except IndexError:
            sys.exit("You are not supposed to be there!")

        action_y, action_x = self.__action_dict[action]
        next_x, next_y = x + action_x, y + action_y

        if (
            7 < next_x or
            0 > next_x or
            7 < next_y or
            0 > next_y
        ):
            return current_state

        if self.__maze[next_y, next_x]:
            next_state = tuple([next_y, next_x])
            return next_state
        else:
            return current_state

    def get_start_state(self):
        return self.__start_state

    def get_goal_state(self):
        return self.__goal_state

    def if_state_legal(self, state):
        y, x = list(state)
        if self.__maze[y, x]:
            return True
        else:
            return False

    def is_goal_state(self, state):
        if state == self.__goal_state:
            return True
        else:
            return False

    def trans_prob_func(
            self,
            state,
            action):
        trans_prob_vec = np.zeros(5)
        next_state = self.perform_action(state, action)
        if next_state == state:
            trans_prob_vec[4] = 1
            return trans_prob_vec
        index = self.__action_space.index(action)
        trans_prob_vec[index] = 1
        return trans_prob_vec
