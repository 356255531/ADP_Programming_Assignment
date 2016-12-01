import numpy as np
from copy import deepcopy

from DP_AlgorithmSyn import DP_AlgorithmSyn


class ValueIterationSyn(DP_AlgorithmSyn):
    """
        Synchronous value iteration DP algortihm
    """

    def __init__(
        self,
        env,
        state_action_space,
        reward,
        alpha,
        epsilon
    ):
        super(ValueIterationSyn, self).__init__()
        self.__env = env
        self.__state_action_space = state_action_space
        self.__reward = reward
        self.__alpha = alpha
        self.__epsilon = epsilon
        self.__val_func_vector = self.__init_val_func_vector(
            state_action_space
        )
        self.__error = []

    def __init_val_func_vector(self, state_action_space):
        val_func_vector = super(ValueIterationSyn, self).init_val_func_vector(
            state_action_space
        )
        val_func_vector_copy = deepcopy(val_func_vector)
        return val_func_vector_copy

    def __cal_trans_prob_mat_and_reward_vector(self, action_sets):
        trans_prob_mat, reward = super(ValueIterationSyn, self).cal_trans_prob_mat_and_reward_vector(
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
        policy = super(ValueIterationSyn, self).derive_policy(
            self.__val_func_vector,
            self.__state_action_space,
            self.__env
        )
        policy_copy = deepcopy(policy)
        return policy_copy

    def get_error(self):
        error = deepcopy(self.__error)
        return error

    def run(self):
        """
            run the value iteration until converged
        """
        error = float("inf")
        count = 0

        action_space = self.__state_action_space.get_action_space()
        num_legal_state = len(self.__state_action_space.get_legal_state_space())
        state_range = [i for i in xrange(0, num_legal_state - 1)]

        while error > self.__epsilon or count < 20:
            pre_val_func_vector = deepcopy(self.__val_func_vector)

            val_func_mat = np.array([])
            for action in action_space:
                action_sets = [action for i in xrange(0, num_legal_state + 1)]
                trans_prob_mat, reward_vector = self.__cal_trans_prob_mat_and_reward_vector(
                    action_sets
                )

                val_func_vector_temp = reward_vector + self.__alpha * np.matmul(
                    trans_prob_mat,
                    self.__val_func_vector
                )

                val_func_mat = np.append(
                    val_func_mat,
                    val_func_vector_temp
                )

            val_func_mat = np.mat(np.reshape(
                val_func_mat,
                (num_legal_state, len(action_space)),
                order='F'
            )
            )

            self.__val_func_vector[state_range, :] = val_func_mat.max(1)[state_range, :]

            error = np.linalg.norm(
                pre_val_func_vector -
                self.__val_func_vector
            )
            if error < self.__epsilon:
                count += 1
            else:
                count = 0
            self.__error.append(error)
