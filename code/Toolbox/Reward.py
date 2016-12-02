class Reward2(object):
    """
        Reward rule 2:
            when goal state reached, given reward 1,
            otherwise punished by -1

        Member function:
            get_reward(current_state, action, next_state):
    """
    def __init__(self, env):
        super(Reward2, self).__init__()
        self.__env = env

    def get_reward(
        self,
        current_state,
        action,
        next_state
    ):
        """
            return reward by given states and action, int
        """
        if self.__env.is_goal_state(next_state):
            return 1
        else:
            return -1


class Reward1(object):
    """
        Reward rule 1:
            when goal state reached, given reward 1,
            otherwise 0

        Member function:
            get_reward(current_state, action, next_state):
    """

    def __init__(self, env):
        super(Reward1, self).__init__()
        self.__env = env

    def get_reward(
        self,
        current_state,
        action,
        next_state
    ):
        """
            return reward by given states and action, int
        """
        if self.__env.is_goal_state(next_state):
            return 1
        else:
            return 0


if __name__ == "__main__":
    print Reward1.__name__
    print Reward2.__name__
