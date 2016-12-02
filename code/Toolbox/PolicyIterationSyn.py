import numpy as np
from copy import deepcopy

from DP_AlgorithmSyn import DP_AlgorithmSyn


class PolicyIterationSyn(DP_AlgorithmSyn):
    """
        Synchronous policy iteration DP algortihm

        Usage:
            simplely call the member function run() after initalization

        Member function:
            __init_val_func_vector(state_action_space)

            __cal_trans_prob_mat_and_reward_vector(action_sets)

            get_val_func_vector()

            get_policy()

            get_error()

            run()

            __policy_evaluation()

            __policy_improvement()

            __if_policy_diff(policy)

        Attributes:
            __env::
                Enviroment class instance

            __state_action_space:
                StateActionSpace class instance

            __reward:
                Reward class instance

            __alpha:
                Learning rate(discount factor), float

            __epsilon:
                Convergence error threashold, float

            __val_func_vector:
                A column vector of value functions of legal states, np.mat

            __policy:
                Learned policy, list of actions, list

            __error:
                Learning error sequence of value iteration, list
    """

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
        """
            derive the value functions of legal states and vectorize into column vector

            return np.mat
        """
        val_func_vector = super(PolicyIterationSyn, self).init_val_func_vector(
            state_action_space
        )
        val_func_vector_copy = deepcopy(val_func_vector)
        return val_func_vector_copy

    def __cal_trans_prob_mat_and_reward_vector(self, action_sets):
        """
            Caculate the transition probility matrix and
            reward function vector of all states with given
            action

            return trans_prob_mat(numpy.mat), reward_vector(numpy.mat)

        """
        trans_prob_mat, reward = super(PolicyIterationSyn, self).cal_trans_prob_mat_and_reward_vector(
            action_sets,
            self.__reward,
            self.__env,
            self.__state_action_space
        )
        return trans_prob_mat, reward

    def get_val_func_vector(self):
        """
            return the value function vector

            return np.mat
        """
        val_func_vector = deepcopy(self.__val_func_vector)
        return val_func_vector

    def get_policy(self):
        """
            return learned policy

            return list
        """
        policy = deepcopy(self.__policy)
        return policy

    def get_error(self):
        """
            return learning error sequence during policy improvement

            return list
        """
        error = deepcopy(self.__error)
        return error

    def run(self):
        """
            run the policy iteration until converged
        """
        pre_policy = deepcopy(self.__policy)
        self.__policy_improvement()
        count = 0

        while self.__if_policy_diff(pre_policy) or count < 5:
            pre_policy = deepcopy(self.__policy)
            self.__policy_evaluation()
            self.__policy_improvement()

            if not self.__if_policy_diff(pre_policy):
                count += 1
            else:
                count = 0

    def __policy_evaluation(self):
        """
            run the policy evaluation until value function converged
        """
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
        """
            do policy improvement
        """
        self.__policy = super(PolicyIterationSyn, self).derive_policy(
            self.__val_func_vector,
            self.__state_action_space,
            self.__env
        )

    def __if_policy_diff(self, policy):
        """
            check whether the policy converged
        """
        isinstance(policy, list)
        if not len(policy) == len(self.__policy):
            return True

        for action1, action2 in zip(self.__policy, policy):
            if action1 != action2:
                return True

        return False


if __name__ == "__main__":
    print PolicyIterationSyn.__name__
