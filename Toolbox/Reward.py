class Reward2(object):
    """docstring for reward"""

    def __init__(self, env):
        super(Reward2, self).__init__()
        self.__env = env

    def get_reward(
        self,
        current_state,
        action,
        next_state
    ):
        if self.__env.is_goal_state(next_state):
            return 1
        else:
            return -1


class Reward1(object):
    """docstring for reward"""

    def __init__(self, env):
        super(Reward1, self).__init__()
        self.__env = env

    def get_reward(
        self,
        current_state,
        action,
        next_state
    ):
        if self.__env.is_goal_state(next_state):
            return 1
        else:
            return 0


if __name__ == "__main__":
    print Reward1.__name__
    print Reward2.__name__
