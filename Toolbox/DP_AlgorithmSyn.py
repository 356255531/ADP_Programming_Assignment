import random as rd


class DP_Algorithm(object):
    """docstring for DP_Algorithm"""

    def init_val_func(self, env):
        # val_func_vector = env.init_val_func()
        # return val_func_vector
        val_func = dict()
        state_space = env.get_state_space()
        for state in state_space:
            if not env.if_state_legal(state):
                val_func[state] = -float("inf")
            else:
                if env.is_goal_state(state):
                    val_func[state] = 10e10
                else:
                    val_func[state] = rd.random()
        return val_func

    def vectorize_val_func(self, val_func):
        pass

    def derive_policy(self, val_func, env):
        policy = dict()
        state_space = env.get_state_space()
        action_space = env.get_action_space()

        for state in state_space:

            if (
                env.is_goal_state(state) or
                not env.if_state_legal(state)
            ):
                continue

            max_val = -float("inf")

            for action in action_space:
                next_state = env.perform_action(state, action)
                if val_func[next_state] > max_val:
                    max_val = val_func[next_state]
                    policy[state] = action
        return policy
