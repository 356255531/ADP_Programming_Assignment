import numpy as np
import sys
import random as rd


class DP_AlgorithmSyn(object):
    """
        Parent class of DP algorithm(VI, PI)
        Member function:
            init_val_func_vector(val_func)

            derive_policy(val_func_vector, env, state_action_space)

            cal_trans_prob_mat_and_reward_vector(
                action,
                reward_func,
                env,
                state_action_space
        )
    """

    def init_val_func_vector(self, state_action_space):
        num_legal_ele = len(state_action_space.get_legal_state_space())
        val_func_vector = np.random.random(num_legal_ele)
        val_func_vector[-1] = 100
        val_func_vector = np.mat(val_func_vector).transpose()
        return val_func_vector

    def derive_policy(
            self,
            val_func_vector,
            state_action_space,
            env
    ):
        """
            Derive the policy from given vectorized value function
            and legal state space
        """
        policy = dict()
        legal_state_space = state_action_space.get_legal_state_space()
        action_space = state_action_space.get_action_space()

        for state in legal_state_space:

            max_val = -float("inf")

            for action in action_space:
                next_state = env.perform_action(state, action)
                feature_vector = env.get_feature_vector_of_legal_state(next_state).transpose()
                val_func = np.matmul(feature_vector, val_func_vector)
                if val_func > max_val:
                    max_val = val_func
                    policy[state] = action
        return policy

    def cal_trans_prob_mat_and_reward_vector(
        self,
        action,
        reward_func,
        env,
        state_action_space
    ):

        legal_state_space = state_action_space.get_legal_state_space()

        state = legal_state_space[0]
        next_state = env.perform_action(
            state,
            action
        )
        feature_vector = state_action_space.get_feature_vector_of_legal_state(next_state)
        feature_vector = feature_vector.transpose()
        trans_prob_mat = np.mat(feature_vector)

        reward = reward_func.get_reward(state, action, next_state)
        reward_vector = np.mat(np.mat(reward))

        for state in legal_state_space:
            next_state = env.perform_action(state, action)

            feature_vector = state_action_space.get_feature_vector_of_legal_state(next_state)
            feature_vector = feature_vector.transpose()
            trans_prob_mat = np.append(
                trans_prob_mat,
                np.mat(feature_vector),
                axis=0
            )

            reward = reward_func.get_reward(state, action, next_state)
            reward_vector = np.append(
                reward_vector,
                np.mat(reward),
                axis=0)

        trans_prob_mat = np.delete(trans_prob_mat, (0), axis=0)
        reward_vector = np.delete(reward_vector, (0), axis=0)
        return trans_prob_mat, reward_vector


if __name__ == "__main__":
    print np.mat(np.random.random(5)).transpose().shape
    print np.matmul(np.zeros(5).transpose(), np.zeros(5))
