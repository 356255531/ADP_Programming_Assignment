import sys


class Agent(object):
    """docstring for Agent"""

    def __init__(self,
                 env
                 ):
        super(Agent, self).__init__()
        self.__previous_state = tuple([-1, -1])
        self.__env = env
        self.__current_state = self.__env.get_start_state()
        self.__state_space = self.__env.get_state_space()
        self.__action_space = self.__env.get_action_space()

    def take_action(self, action):
        try:
            if action not in self.__action_space:
                raise ValueError
        except ValueError:
            sys.exit("Action illeagal, stay around.\n")

        temp_state = self.__env.perform_action(
            self.__current_state, action
        )
        if temp_state == self.__current_state:
            self.__previous_state = temp_state
            print "Still in state", temp_state, "with action", action, "\n"
        else:
            print "Move to state", temp_state, "with action", action, "\n"
            self.__previous_state = self.__current_state
            self.__current_state = temp_state

    def get_previous_state(self):
        print "The previous state is", self.__previous_state
        return self.__previous_state

    def get_current_state(self):
        print "The current state is", self.__current_state
        return self.__current_state

    def if_in_goal_state(self):
        if self.__env.is_goal_state(
            self.__current_state
        ):
            return True
        else:
            return False

    def if_goal_state(self, state):
        if self.__env.is_goal_state(state):
            return True
        else:
            return False


if __name__ == "__main__":
    pass
