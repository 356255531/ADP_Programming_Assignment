import numpy as np
from StateActionSpace import StateActionSpace
from Enviroment import Enviroment


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
        """
            Randomly initialize the value function
            vector and assgin the value of goal state
            to 100 at the end of value function vector

            return numpy.mat
        """
        isinstance(state_action_space, StateActionSpace)
        num_legal_ele = len(state_action_space.get_legal_state_space())
        val_func_vector = np.random.random(num_legal_ele)
        val_func_vector[-1] = 0
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
        policy = []

        legal_state_space = state_action_space.get_legal_state_space()
        action_space = state_action_space.get_action_space()

        for state in legal_state_space:

            max_val = -float("inf")

            for action in action_space:
                next_state = env.perform_action(state, action)
                feature_vector = state_action_space.get_feature_vector_of_legal_state(next_state).transpose()
                val_func = np.matmul(
                    np.mat(feature_vector),
                    val_func_vector
                )
                if val_func > max_val:
                    max_val = val_func
                    policy_temp = action
            policy.append(policy_temp)

        return policy

    def cal_trans_prob_mat_and_reward_vector(
        self,
        action_sets,
        reward_func,
        env,
        state_action_space
    ):
        """
            Caculate the transition probility matrix and
            reward function vector of all states with given
            action

            return trans_prob_mat(numpy.mat), reward_vector(numpy.mat)
        """
        isinstance(action_sets, list)
        isinstance(env, Enviroment)
        isinstance(state_action_space, StateActionSpace)
        legal_state_space = state_action_space.get_legal_state_space()

        trans_prob_mat, reward_vector = np.array([]), np.array([])
        for state, action in zip(legal_state_space, action_sets):
            next_state = env.perform_action(state, action)

            feature_vector = state_action_space.get_feature_vector_of_legal_state(next_state)
            trans_prob_mat = np.append(
                trans_prob_mat,
                feature_vector,
            )

            reward = reward_func.get_reward(state, action, next_state)
            reward_vector = np.append(
                reward_vector,
                reward,
            )

        num_legal_state = len(legal_state_space)
        trans_prob_mat = np.mat(np.reshape(trans_prob_mat, (num_legal_state, num_legal_state)))
        reward_vector = np.mat(np.reshape(reward_vector, (num_legal_state, 1)))
        return trans_prob_mat, reward_vector


if __name__ == "__main__":
    a = []
    print np.array(a)
