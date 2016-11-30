from copy import deepcopy
import numpy as np

from DP_Algorithm import DP_Algorithm


class PolicyIterationAsyn(DP_Algorithm):
    """docstring for PolicyIterationAsyn"""

    def __init__(
        self,
        env,
        reward,
        alpha,
        epsilon
    ):
        super(PolicyIterationAsyn, self).__init__()
        self.__env = env
        self.__reward = reward
        self.__alpha = alpha
        self.__epsilon = epsilon
        self.__val_func = self.__init_val_func()
        self.__policy = dict()

    def __init_val_func(self):
        val_func = super(PolicyIterationAsyn, self).init_val_func(self.__env)
        return val_func

    def get_val_func(self):
        val_func = deepcopy(self.__val_func)
        return val_func

    def get_policy(self):
        policy = deepcopy(self.__policy)
        return policy

    def print_val_func(self):
        for y in xrange(0, 8):
            temp = [round(self.__val_func[(y, x)], 2) for x in xrange(0, 8)]
            print temp
        # print self.__val_func

    def build_states_val_vector(self, state):
        round_val_vector = np.zeros(5)
        round_val_vector[4] = self.__val_func[state]
        action_space = self.__env.get_action_space()
        next_states = [self.__env.perform_action(state, i) for i in action_space]
        for next_state in next_states:
            round_val_vector[next_states.index(next_state)] = self.__val_func[next_state]
        return round_val_vector

    def run(self):
        pre_policy = deepcopy(self.__policy)
        self.__policy_improvement()

        while self.__if_policy_diff(pre_policy):
            pre_policy = deepcopy(self.__policy)
            self.__policy_evaluation()
            self.__policy_improvement()

    def __policy_evaluation(self):
        diff = float("inf")
        state_space = self.__env.get_state_space()

        val_func = []
        for i in self.__val_func.values():
            if i != -float("inf"):
                val_func.append(i)

        while diff > self.__epsilon:
            pre_val_func = val_func

            for state in state_space:
                if (
                    self.__env.is_goal_state(state) or
                    not self.__env.if_state_legal(state)
                ):
                    continue

                action = self.__policy[state]
                next_state = self.__env.perform_action(state, action)
                step_reward = self.__reward.get_reward(
                    state,
                    action,
                    next_state
                )
                self.__val_func[state] = step_reward + self.__alpha * np.dot(
                    self.__env.trans_prob_func(state, action),
                    self.build_states_val_vector(state)
                )

            val_func = []
            for i in self.__val_func.values():
                if i != -float("inf"):
                    val_func.append(i)

            diff = np.linalg.norm(
                np.array(pre_val_func) -
                np.array(val_func)
            )

    def __policy_improvement(self):
        self.__policy = super(PolicyIterationAsyn, self).derive_policy(
            self.__val_func,
            self.__env
        )

    def __if_policy_diff(self, policy):
        if set(self.__policy.keys()) != set(policy.keys()):
            return True

        state_in_policy = self.__policy.keys()

        for state in state_in_policy:
            if self.__policy[state] != policy[state]:
                return True

        return False


if __name__ == "__main__":
    print PolicyIterationAsyn.__name__
