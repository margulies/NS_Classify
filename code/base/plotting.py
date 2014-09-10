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


def heat_map(data, x_labels, y_labels, size = (11, 11), file_name=None, lines=None, lines_off = 0):

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.YlOrRd, alpha=0.8)

    fig = plt.gcf()

    fig.set_size_inches(size)

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

    if lines is not None:
        xl, xh=ax.get_xlim()
        yl, yh=ax.get_ylim()
        if lines_off is not None:
            yl -= lines_off
            xh -= lines_off
        ax.hlines(lines, xl, xh, color='w', linewidth = 1.5)
        ax.vlines(lines, yl, yh, color='w', linewidth = 1.5)

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

def plot_roc(y_test, y_pred):
    import pylab as pl
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)

    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show()


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
