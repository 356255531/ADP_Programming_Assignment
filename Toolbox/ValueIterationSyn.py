import numpy as np
from copy import deepcopy

from DP_Algorithm import DP_Algorithm


class ValueIterationAsyn(DP_Algorithm):
    """docstring for ValueIterationAsyn"""

    def __init__(
        self,
        env,
        reward,
        alpha,
        epsilon
    ):
        super(ValueIterationAsyn, self).__init__()
        self.__env = env
        self.__reward = reward
        self.__alpha = alpha
        self.__epsilon = epsilon
        self.__val_func = self.__init_val_func()

    def __init_val_func(self):
        val_func_vector = super(ValueIterationAsyn, self).init_val_func(self.__env)
        return val_func_vector

    def get_val_func(self):
        val_func_vector = deepcopy(self.__val_func)
        return val_func_vector

    def print_val_func(self):
        for x in xrange(0, 8):
            temp = [round(self.__val_func[(y, x)], 2) for y in xrange(0, 8)]
            print temp

    def get_policy(self):
        policy = super(ValueIterationAsyn, self).derive_policy(
            self.__val_func,
            self.__env
        )
        return policy

    def build_states_val_vector(self, state):
        round_val_vector = np.zeros(5)
        round_val_vector[4] = self.__val_func[state]
        action_space = self.__env.get_action_space()
        next_states = [self.__env.perform_action(state, i) for i in action_space]
        for next_state in next_states:
            round_val_vector[next_states.index(next_state)] = self.__val_func[next_state]
        return round_val_vector

    def run(self):
        diff = float("inf")
        state_space = self.__env.get_state_space()
        action_space = self.__env.get_action_space()

        val_func_vector = []
        for i in self.__val_func.values():
            if i != -float("inf"):
                val_func_vector.append(i)

        while diff > self.__epsilon:
            pre_val_func = val_func_vector

            for state in state_space:
                if (
                    self.__env.is_goal_state(state) or
                    not self.__env.if_state_legal(state)
                ):
                    continue
                val = []
                for action in action_space:
                    next_state = self.__env.perform_action(state, action)
                    step_reward = self.__reward.get_reward(
                        state,
                        action,
                        next_state
                    )
                    val.append(step_reward + self.__alpha * np.dot(
                        self.__env.trans_prob_func(state, action),
                        self.build_states_val_vector(state)
                    )
                    )
                self.__val_func[state] = max(val)

            val_func_vector = []
            for i in self.__val_func.values():
                if i != -float("inf"):
                    val_func_vector.append(i)

            diff = np.linalg.norm(
                np.array(pre_val_func) -
                np.array(val_func_vector)
            )
