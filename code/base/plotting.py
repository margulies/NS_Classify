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

    fig.set_size_inches(8, 11)

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


def region_heatmap(self, basename=None, zscore_regions=False, zscore_features=False, thresh=None, subset=None, each_region=True):
    """" Makes a heatmap of the importances of the classification. Makes an overall average heatmap
    as well as a heatmap for each individual region. Optionally, you can specify the heatmap to be
    z-scored. You can also specify a threshold.

    Args:
        basename: string, base directory and file name
        zscore_regions: boolean, should heatmap be z-scored based within regions
        zscore_regions: boolean, should heatmap be z-scored based within features
        thresh: value to threshold heatmap. Only values above this value are kept
        subset: what regions should be plotted? default is all

    Outputs:
        Outputs a .png file for the overall heatmap and for each region. If z-scored on thresholded,
        will denote in file name using z0 (regions), z1 (features), and/or t followed by threshold.
    """

    from plotting import heat_map

    if subset is None:
        subset = range(0, self.mask_num)

    overall_fi = self.feature_importances[subset][:, subset]
    if np.array(subset).max() > self.mask_num:
        print "Warning: you entered an incorrect mask index!"

    fi = overall_fi.mean(axis=0).T

    z0 = ""
    z1 = ""
    t = ""

    if zscore_regions:
        fi = np.apply_along_axis(stats.zscore, 0, fi)
        z0 = "z0_"
    if zscore_features:
        fi = np.apply_along_axis(stats.zscore, 1, fi)
        z1 = "z1_"

    if thresh is not None:
        fi = np.ma.masked_array(fi)
        fi.mask = fi < thresh
        t = "zt" + str(thresh) + "_"

    heat_map(fi, np.array(subset) + 1, self.feature_names,
             basename + "imps_hm_" + z0 + z1 + t + "overall.png")

    if each_region:
        for i in subset:

            fi = overall_fi[subset.index(i)].T

            if zscore_regions:
                fi = np.ma.masked_invalid(stats.zscore(fi, axis=0))
            if zscore_features:
                fi = stats.zscore(fi, axis=1)

            if thresh is not None:
                fi.mask = fi < thresh

            if basename is None:
                file_name = None
            else:
                file_name = basename + "imps_hm_" + \
                    z0 + z1 + t + str(i) + ".png"

            heat_map(fi, np.array(subset) + 1, self.feature_names, file_name)

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
