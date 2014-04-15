#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def density_plot(data, file_name=None, covariance_factor=.2):
    """ Generate a density plot """
    data = np.array(data)
    density = stats.gaussian_kde(data)
    xs = np.linspace(0, data.max() + data.max() / 10, 200)
    density.covariance_factor = lambda: covariance_factor
    density._compute_covariance()
    plt.plot(xs, density(xs))

    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)


def heat_map(data, x_labels, y_labels, file_name=None):

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.YlOrRd, alpha=0.8)

    fig = plt.gcf()

    fig.set_size_inches(11, 11)

    # turn off the frame
    ax.set_frame_on(False)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_yticklabels(y_labels, minor=False)
    ax.set_xticklabels(x_labels, minor=False)

    ax.grid(False)

    # Turn off all the ticks
    ax = plt.gca()

    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    if file_name is None:
        fig.show()
    else:
        fig.savefig(file_name)


def save_region_importance_plots(clf, basename, thresh=20):
    for i in range(1, clf.mask_num):
        clf.plot_importances(
            i - 1, file_name=basename + "_imps_" + str(i) + ".png", thresh=thresh)
        clf.plot_importances(
            None, file_name=basename + "_imps_overall.png", thresh=thresh)


# def plot_min_max_fi(clf):
#     density_plot(
#         clf.feature_importances[tuple(np.where(sh_1 == sh_1.min())[0])],
#         file_name="../results/diagonstic/sh_1_min.png")
#     density_plot(
#         clf.feature_importances[tuple(np.where(sh_1 == sh_1.max())[0])],
#         file_name="../results/diagonstic/sh_1_max.png")
#     density_plot(clf.feature_importances[np.where(sh == sh.max())[0]][
#                  :, np.where(sh == sh.max())[1]], file_name="../results/diagonstic/sh_0_max.png")
#     density_plot(clf.feature_importances[np.where(sh == sh.min())[0]][
#                  :, np.where(sh == sh.min())[1]], file_name="../results/diagonstic/sh_0_min.png")
