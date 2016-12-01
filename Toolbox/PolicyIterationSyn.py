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
        self.__policy = []
        self.__error = []

    def __init_val_func_vector(self, state_action_space):
        val_func_vector = super(PolicyIterationSyn, self).init_val_func_vector(
            state_action_space
        )
        val_func_vector_copy = deepcopy(val_func_vector)
        return val_func_vector_copy

    def __cal_trans_prob_mat_and_reward_vector(self, action_sets):
        trans_prob_mat, reward = super(PolicyIterationSyn, self).cal_trans_prob_mat_and_reward_vector(
            action_sets,
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

    def get_error(self):
        error = deepcopy(self.__error)
        return error

    def run(self):
        pre_policy = deepcopy(self.__policy)
        self.__policy_improvement()

        while self.__if_policy_diff(pre_policy):
            pre_policy = deepcopy(self.__policy)
            self.__policy_evaluation()
            self.__policy_improvement()

    def __policy_evaluation(self):
        error = float("inf")
        count = 0

        num_legal_state = len(self.__state_action_space.get_legal_state_space())
        state_range = [i for i in xrange(0, num_legal_state - 1)]

        while error > self.__epsilon or count < 20:
            pre_val_func_vector = deepcopy(self.__val_func_vector)

            trans_prob_mat, reward_vector = self.__cal_trans_prob_mat_and_reward_vector(
                self.__policy
            )

            val_func_vector_temp = reward_vector + self.__alpha * np.matmul(
                trans_prob_mat,
                self.__val_func_vector
            )

            self.__val_func_vector[state_range, :] = val_func_vector_temp[state_range, :]

            error = np.linalg.norm(
                pre_val_func_vector -
                self.__val_func_vector
            )
            if error < self.__epsilon:
                count += 1
            else:
                count = 0
            self.__error.append(error)

    def __policy_improvement(self):
        self.__policy = super(PolicyIterationSyn, self).derive_policy(
            self.__val_func_vector,
            self.__state_action_space,
            self.__env
        )

    def __if_policy_diff(self, policy):
        isinstance(policy, list)
        if not len(policy) == len(self.__policy):
            return True

        for action1, action2 in zip(self.__policy, policy):
            if action1 != action2:
                return True

        return False


if __name__ == "__main__":
    print PolicyIterationSyn.__name__
