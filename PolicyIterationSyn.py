import numpy as np
from copy import deepcopy

from DP_AlgorithmSyn import DP_AlgorithmSyn


class PolicyIterationSyn(DP_AlgorithmSyn):
    """docstring for PolicyIterationSyn"""

    def __init__(
        self,
        env,
        state_action_space,
        reward,
        alpha,
        epsilon
    ):
        super(PolicyIterationSyn, self).__init__()
        self.__env = env
        self.__state_action_space = state_action_space
        self.__reward = reward
        self.__alpha = alpha
        self.__epsilon = epsilon
        self.__val_func_vector = self.__init_val_func_vector(
            state_action_space
        )
        self.__policy = list()

    def __init_val_func_vector(self, state_action_space):
        val_func_vector = super(PolicyIterationSyn, self).init_val_func_vector(
            state_action_space
        )
        val_func_vector_copy = deepcopy(val_func_vector)
        return val_func_vector_copy

    def __cal_trans_prob_mat_and_reward_vector(self, action):
        trans_prob_mat, reward = super(PolicyIterationSyn, self).cal_trans_prob_mat_and_reward_vector(
            action,
            self.__reward,
            self.__env,
            self.__state_action_space
        )
        return trans_prob_mat, reward

    def get_val_func_vector(self):
        val_func_vector = deepcopy(self.__val_func_vector)
        return val_func_vector

    def get_policy(self):
        policy = deepcopy(self.__policy)
        return policy

    def run(self):
        pre_policy = deepcopy(self.__policy)
        self.__policy_improvement()

        while self.__if_policy_diff(pre_policy):
            pre_policy = deepcopy(self.__policy)
            self.__policy_evaluation()
            self.__policy_improvement()

    def __policy_evaluation(self):
        diff = float("inf")

        while diff > self.__epsilon:
            pre_val_func_vector = deepcopy(self.__val_func_vector)

            num_legal_state = len(self.__state_action_space.get_legal_state_space())
            state_range = [i for i in xrange(0, num_legal_state - 1)]

            val_func_mat = np.array([])

            self.__val_func_vector[state_range, :] = val_func_mat.max(1)[state_range, :]

            diff = np.linalg.norm(
                pre_val_func_vector -
                self.__val_func_vector
            )

    def __policy_improvement(self):
        self.__policy = super(PolicyIterationSyn, self).derive_policy(
            self.__val_func,
            self.__env
        )

    def __if_policy_diff(self, policy):
        if not len(policy) == len(self.__policy):
            return False

        for action1, action2 in zip(self.__policy, policy):
            if action1 != action2:
                return False

        return True


if __name__ == "__main__":
    print PolicyIterationSyn.__name__
