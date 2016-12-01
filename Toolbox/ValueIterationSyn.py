import numpy as np
import sys
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
        self.__val_func_vector = self.__init_val_func_vector(state_action_space)

    def __init_val_func_vector(self, state_action_space):
        val_func_vector = super(ValueIterationSyn, self).init_val_func_vector(
            state_action_space
        )
        val_func_vector_copy = deepcopy(val_func_vector)
        return val_func_vector_copy

    def __cal_trans_prob_mat_and_reward_vector(self, action):
        trans_prob_mat, reward = super(ValueIterationSyn, self).cal_trans_prob_mat_and_reward_vector(
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
        policy = super(ValueIterationSyn, self).derive_policy(
            self.__val_func_vector,
            self.__state_action_space,
            self.__env
        )
        policy_copy = deepcopy(policy)
        return policy_copy

    def run(self):
        diff = float("inf")

        action_space = self.__state_action_space.get_action_space()

        while diff > self.__epsilon:
            pre_val_func_vector = deepcopy(self.__val_func_vector)

            trans_prob_mat, reward_vector = self.__cal_trans_prob_mat_and_reward_vector(
                action_space[0]
            )
            val_func_vector_temp = reward_vector + np.matmul(
                trans_prob_mat,
                self.__val_func_vector
            )

            row_num, _ = val_func_vector_temp.shape
            val_func_vector_temp = val_func_vector_temp[[0, row_num - 2], :]
            val_func_mat = np.mat(val_func_vector_temp)

            for action in action_space:
                trans_prob_mat, reward_vector = self.__cal_trans_prob_mat_and_reward_vector(
                    action
                )

                val_func_vector_temp = reward_vector + np.matmul(
                    trans_prob_mat,
                    self.__val_func_vector
                )

                val_func_vector_temp = val_func_vector_temp[[0, row_num - 2], :]

                val_func_mat = np.append(
                    val_func_mat,
                    val_func_vector_temp,
                    axis=1
                )
            val_func_mat = np.delete(
                val_func_mat,
                0,
                axis=1
            )
            # print val_func_mat
            print val_func_mat.max(1)
            # sys.exit(0)
            self.__val_func_vector[[0, row_num - 2], :] = val_func_mat.max(1)
            # print self.__val_func_vector
            diff = np.linalg.norm(
                pre_val_func_vector -
                self.__val_func_vector
            )
        print self.get_val_func_vector().ravel()
