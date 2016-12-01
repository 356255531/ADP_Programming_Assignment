import matplotlib.pyplot as plt
import numpy as np


class PlotAgent(object):
    """docstring for PlotAgent"""

    def __init__(self):
        super(PlotAgent, self).__init__()

    def plot_policies(
        self,
        plot_policies,
        plot_labels
    ):
        pass

    def plot_error(
        self,
        error_sets,
        plot_labels
    ):
        plt.figure(figsize=(15.0, 15.0))
        for num in xrange(0, len(error_sets)):

            t = np.arange(len(error_sets[num]))
            image_index = num + 1
            plt.subplot(4, 3, image_index)
            plt.plot(t, error_sets[num])
            plt.title(plot_labels[num])

        plt.savefig('ErrorPlot.jpg')


if __name__ == '__main__':
    print int("25")
