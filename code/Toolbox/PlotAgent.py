import matplotlib.pyplot as plt
import numpy as np


class PlotAgent(object):
    """docstring for PlotAgent"""

    def __init__(
        self,
        env,
        state_action_space
    ):
        super(PlotAgent, self).__init__()
        self.__env = env
        self.__state_action_space = state_action_space

    def plot_policies(
        self,
        plot_policies,
        plot_labels
    ):
        plt.figure(figsize=(20.0, 20.0))
        action_dict = self.__env.get_action_dict()
        for num in xrange(0, len(plot_policies)):
            soaList = []
            state_space = self.__state_action_space.get_state_space()
            for state in state_space:
                if not self.__env.if_state_legal(state) or self.__env.is_goal_state(state):
                    continue
                state_index = np.where(
                    self.__state_action_space.get_feature_vector_of_legal_state(state) == 1
                )[0][0]
                action = plot_policies[num][state_index]
                action_y_x = action_dict[action]
                y, x = state
                n, m = action_y_x
                soaList.append([x, y, m, n])
            X, Y, U, V = zip(*soaList)

            image_index = num + 1
            plt.subplot(4, 4, image_index)
            plt.title(plot_labels[num])
            ax = plt.gca()
            for state in state_space:
                if not self.__env.if_state_legal(state):
                    y, x = state
                    plt.plot(x, y, 'bs')
            plt.plot(6, 6, 'ro')
            ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
            ax.set_xlim([-1, 8])
            ax.set_ylim([8, -1])

        plt.savefig('PolicyPlot.jpg')

    def plot_error(
        self,
        error_sets,
        plot_labels
    ):
        plt.figure(figsize=(20.0, 20.0))
        for num in xrange(0, len(error_sets)):

            t = np.arange(len(error_sets[num]))
            image_index = num + 1
            plt.subplot(4, 4, image_index)
            plt.plot(t, error_sets[num])
            plt.title(plot_labels[num])

        plt.savefig('ErrorPlot.jpg')


if __name__ == '__main__':
    print int("25")
