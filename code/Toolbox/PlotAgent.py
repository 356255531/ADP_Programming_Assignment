import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

__auther__ = "Zhiwei"


class PlotAgent(object):
    """
        This class plot different kinds of image of algorithm analysis

        Member function:
            plot_policies(plot_policies,plot_labels)

            plot_error(error_sets, plot_labels)

            plot_val_func(val_func_vector_sets, plot_labels, state_action_space, env)
    """

    def __init__(
        self,
        env,
        state_action_space,
    ):
        super(PlotAgent, self).__init__()
        self.__env = env
        self.__state_action_space = state_action_space

    def plot_policies(
        self,
        plot_policies,
        plot_labels,
        file_name='PolicyPlot.jpg'
    ):
        """
            visualize the given policies and save the image

            no return
        """
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

        plt.savefig(file_name)

    def plot_error(
        self,
        error_sets,
        plot_labels,
        file_name='ErrorPlot.jpg'
    ):
        """
            plots the given errors and save the image

            no return
        """
        plt.figure(figsize=(20.0, 20.0))
        for num in xrange(0, len(error_sets)):

            t = np.arange(len(error_sets[num]))
            image_index = num + 1
            plt.subplot(4, 4, image_index)
            plt.plot(t, error_sets[num])
            plt.title(plot_labels[num])

        plt.savefig(file_name)

    def plot_val_func(
        self,
        val_func_vector_sets,
        plot_labels,
        state_action_space,
        env,
        file_name='ValueFunction.jpg'
    ):
        """
            visualize the given value functions with color boxes and save the image

            no return
        """
        plt.figure(figsize=(20.0, 20.0))

        for num in xrange(0, len(val_func_vector_sets)):
            image_index = num + 1
            plt.subplot(4, 4, image_index)
            plt.title(plot_labels[num])

            rectangle_list = []
            state_space = env.get_state_space()
            for state in state_space:
                y, x = state
                if env.if_state_legal(state):
                    alpha_color = np.log10(
                        np.matmul(
                            state_action_space.get_feature_vector_of_legal_state(state).transpose(),
                            val_func_vector_sets[num]) + 5
                    ) / np.log10(25)
                else:
                    alpha_color = 0
                rectangle_list.append(
                    patches.Rectangle(
                        (x, y), 1, 1,
                        alpha=alpha_color
                    )
                )
            for p in rectangle_list:
                plt.subplot(4, 4, image_index).add_patch(p)

            plt.gca().set_xlim([-1, 8])
            plt.gca().set_ylim([8, -1])

        plt.savefig(file_name)


if __name__ == '__main__':

    origin = 'lower'
    #origin = 'upper'

    delta = 1

    # x = y = np.arange(-3.0, 3.01, delta)
    # x = np.arange(8, -1.01, delta)
    # y = np.arange(8, -1.01, delta)
    x = [-1, 0, 1]
    y = [-1, 0, 1]
    X, Y = np.meshgrid(x, y)
    print X, Y
    Z1 = plt.mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = plt.mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    Z = 10 * (Z1 - Z2)

    nr, nc = Z.shape

    # put NaNs in one corner:
    Z[-nr // 6:, -nc // 6:] = np.nan
    # contourf will convert these to masked

    Z = np.ma.array(Z)
    # mask another corner:
    Z[:nr // 6, :nc // 6] = np.ma.masked

    # mask a circle in the middle:
    interior = np.sqrt((X**2) + (Y**2)) < 0.5
    Z[interior] = np.ma.masked
    # Now make a contour plot with the levels specified,
    # and with the colormap generated automatically from a list
    # of colors.
    levels = [-1.5, -1, -0.5, 0, 0.5, 1]

    # CS3 = plt.contourf(X, Y, Z, levels,
    #                    colors=('r', 'g', 'b'),
    #                    origin=origin,
    #                    extend='both')
    # # Our data range extends outside the range of levels; make
    # # data below the lowest contour level yellow, and above the
    # # highest level cyan:
    # CS3.cmap.set_under('yellow')
    # CS3.cmap.set_over('cyan')

    # CS4 = plt.contour(X, Y, Z, levels,
    #                   colors=('k',),
    #                   linewidths=(3,),
    #                   origin=origin)
    # plt.title('Listed colors (3 masked regions)')
    # plt.clabel(CS4, fmt='%2.1f', colors='w', fontsize=14)

    # Notice that the colorbar command gets all the information it
    # needs from the ContourSet object, CS3.
    # plt.colorbar(CS3)

    # Illustrate all 4 possible "extend" settings:
    extends = ["neither", "both", "min", "max"]
    cmap = plt.cm.get_cmap("winter")
    cmap.set_under("magenta")
    cmap.set_over("yellow")
    # Note: contouring simply excludes masked or nan regions, so
    # instead of using the "bad" colormap value for them, it draws
    # nothing at all in them.  Therefore the following would have
    # no effect:
    # cmap.set_bad("red")
    print X
    print Y
    Z = [[0.585205655172237, 0.9647535023178831, 0.584786441437148],
         [0.8472560269717786, 0, 0.6781331356092241],
         [-0.28690906075747713, 0.7338940176031497, 0]]
    print Z
    fig, axs = plt.subplots(2, 2)
    for ax, extend in zip(axs.ravel(), extends):
        cs = ax.contourf(X, Y, Z, levels, cmap=cmap, extend=extend, origin=origin)
        fig.colorbar(cs, ax=ax, shrink=0.9)
        ax.set_title("extend = %s" % extend)
        ax.locator_params(nbins=4)

    plt.show()

    x = [1, 2, 3, 4]
    y = [1, 2, 3, 4]
    m = [[15, 14, 13, 12], [14, 12, 10, 8], [13, 10, 7, 4], [12, 8, 4, 0]]
    cs = plt.contour(x, y, m, [9.5])
    cs.collections[0].get_paths()
