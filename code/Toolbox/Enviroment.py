from copy import deepcopy
import numpy as np
import itertools
import sys

__auther__ = "Zhiwei"


class Enviroment(object):
    """
    Member function:
        if_state_out_of_range(tuple state)

        if_state_legal(tuple state)

        if_action_legal(string action)

        get_state_space()

        get_start_state()

        get_goal_state()

        is_goal_state(tuple state)

        get_action_space()

        get_action_dict()

        perform_action(current_state, action)

    Attributes:
        __maze:
            maze map, 2D Array, bool,
            available state with true, unavailable state with false

        __state_space:
            all states, illegal or legal, tuple, (y, x)

        __action_space:
            all actions, string

        __action_dict:
            key is actions, value is motion (delta_y, delta_x), tuple

        __start_state:
            (start_y, start_x), tuple

        __goal_state:
            (goal_y, goal_x), tuple

    """

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

    def if_state_out_of_range(self, state):
        """
            judge if the given state is out of range[0-7][0-7]
            if out of region, system exits

            no return
        """
        try:
            if state not in self.__state_space:
                raise ValueError
        except ValueError:
            sys.exit("Can't judge if the state is legal, because it's not in the state space.")
        return True

    def if_state_legal(self, state):
        """
            judge if the given state if illegal(a wall)

            yes return true, otherwise false
        """
        self.if_state_out_of_range(state)

        y, x = list(state)
        if self.__maze[y, x]:
            return True
        else:
            return False

    def if_action_legal(self, action):
        """
            judge if a given action is illegal(not in given action space)
            when illegal system exits

            no return
        """
        try:
            if action not in self.__action_space:
                raise ValueError
        except ValueError:
            sys.exit("Can't perform action, because the action is not in the action space.")
        return True

    def get_state_space(self):
        """
            return the state space(illegal and legal), list
        """
        state_space = deepcopy(self.__state_space)
        return state_space

    def get_start_state(self):
        """
            return the start state, tuple
        """
        return self.__start_state

    def get_goal_state(self):
        """
            return the goal state, tuple
        """
        return self.__goal_state

    def is_goal_state(self, state):
        """
            judge if given state(tuple) is goal state

            return true if yes, otherwise no
        """
        self.if_state_out_of_range(state)
        if state == self.__goal_state:
            return True
        else:
            return False

    def get_action_space(self):
        """
            return the action space, list of string
        """
        action_space = deepcopy(self.__action_space)
        return action_space

    def get_action_dict(self):
        """
            return the action dictionary, key is action(string),
            value is coordinates difference according to action
            (delta_y, delta_x)

            return dict
        """
        action_dict = deepcopy(self.__action_dict)
        return action_dict

    def perform_action(
        self,
        current_state,
        action
    ):
        """
            current_state(tuple), action(string)

            return the coordinate of next state, tuple
        """
        self.if_state_out_of_range(current_state)
        self.if_action_legal(action)
        if not self.if_state_legal(current_state):
            sys.exit("Can't perform action, because agent is not supposed to be in illegal state(wall).")

        y, x = current_state
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
            next_state = (next_y, next_x)
            return next_state
        else:
            return current_state


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
    env
    print env.get_state_space()
    print env.get_action_space()
    print env.get_action_dict()
    print env.get_start_state()
    print env.get_goal_state()
    print env.is_goal_state((0, 1))
    print env.perform_action((6, 3), 'up')
