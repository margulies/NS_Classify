#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from nipype.interfaces import fsl
from scipy import stats

import itertools


from ..statistics import shannons
from .. import tools


def get_ns_for_pairs(a_b_c):
    """ Parallel  wrapper of doing bincount on y """
    X, y = a_b_c

    return np.bincount(y)

def calculate_ns(clf):
    mask_pairs = list(itertools.combinations(clf.masklist, 2))
    clf.ns = np.ma.masked_array(
        np.empty((clf.mask_num, clf.mask_num, 2)), True)

    pb = tools.ProgressBar(len(list(mask_pairs)))

    for index, n in itertools.imap(get_ns_for_pairs, itertools.izip(itertools.repeat(clf.dataset), mask_pairs, itertools.repeat(clf.thresh))):
        clf.ns[index] = n
        pb.next()

def get_mask_averages(clf, precision=None, subset=None):

    if subset is not None:
        final_score = clf.final_score[subset][:, subset]
    else:
        final_score = clf.final_score

    averages = [final_score[k].mean() for k in range(0,
                                                     final_score.shape[0])]

    if precision is not None:
        averages = [round(x, precision) for x in averages]

    return averages

def min_features(clf):  
    return np.array([n_features for n_features in clf.fit_clfs.flatten()]).mean()

def plot_importances(clf, index, thresh=20, file_name=None, absolute=False, ranking=False):
    """ Plot importances for a given index 
    Args:
        index: Can be an tuple index comparing two masks (2, 3),
            an integer index (i.e. average for mask 2),
            or None, which indicates overall average 
        thresh: Minimum importance to plot
        file: Optional string indicating location of file to save plot
            instead of displaying

    Output:
        Either shows plot or saves it
    """

    import pylab as pl

    [imps, names] = zip(
        *clf.get_importances(index, absolute=absolute, ranking=ranking))

    imps = np.array(imps)
    imps = imps[imps > thresh]

    names = np.array(names)
    names = names[-len(imps):]

    sorted_idx = np.argsort(imps)
    pos = np.arange(sorted_idx.shape[0]) + .5
    pl.subplot(1, 2, 2)

    pl.barh(pos, imps[sorted_idx], align='center')
    pl.yticks(pos, names[sorted_idx])
    pl.xlabel('Relative Importance')
    pl.title('Variable Importance')

    if not file_name:
        pl.show()
    else:
        pl.savefig(file_name, bbox_inches=0)
        pl.close()

def get_best_features(clf, n, ranking=True):
    """ Gets the n best features across all comparisons from a RFE classifier """

    return clf.get_importances(None, absolute=True, ranking=ranking)[-n:]


def importance_stats(clf, method='shannons', axis=0, average=True, subset=None):
    """ Returns various statics on the importances for each masks
    These funcions are intended to be used to summarize how consistent or correlated 
    the importance matrices are within each region 

    axis = 0 applies across regions
    shape is len(features)
    axis = 1 is equivalent to applying to within regions

    average: average results within axis of interest?
    subset: Only do for a subset of the data
    """

    if subset is None:
        subset = range(0, clf.mask_num)

    results = []

    fi = clf.feature_importances[subset][:, subset]

    for i in subset:
        region_data = fi[subset.index(i)]

        if method == 'var':
            results.append(np.apply_along_axis(np.var, axis, region_data))

        elif method == 'shannons':
            results.append(
                np.apply_along_axis(shannons, axis, region_data))

    results = np.array(results)

    if axis == 1:
        results = np.ma.masked_array(results)
        i, j = np.meshgrid(*map(np.arange, results.shape), indexing='ij')
        results.mask = (i == j)

    if average:
        return results.mean(axis=1)
    else:
        return results
            
def get_mean_region_importances(clf, subset=None):

    if subset is None:
        subset = range(0, clf.mask_num)

    fi = clf.feature_importances[subset][:, subset]

    results = np.array(fi.mean(axis=2).mean(axis=0))

    return results

def accuracy_stats(clf, method='shannons', subset=None):

    if subset is None:
        subset = range(0, clf.mask_num)

    fs = clf.final_score[subset][:, subset]

    results = []
    for row in subset:
        if method == 'shannons':
            results.append(shannons(fs[subset.index(row)]))
        elif method == 'var':
            results.append(fs[subset.index(row)].var())

    return results

def minN_by_region(clf):
    """ Returns the average N for the smallest class in each comparison for each region """
    results = []
    for i in clf.c_data:
        r = []
        for j in i:
            if j is not None:
                r.append(np.bincount(j[1])[np.bincount(j[1]) != 0].min())

        results.append(np.array(r).mean())
    return results

def region_heatmap(clf, basename=None, zscore_regions=False, zscore_features=False, thresh=None, subset=None, compare=False, region=None):
    """" Makes a heatmap of the importances of the classification. Makes an overall average heatmap
    as well as a heatmap for each individual region. Optionally, you can specify the heatmap to be
    z-scored. You can also specify a threshold.

    Args:
        basename: string, base directory and file name
        zscore_regions: boolean, should heatmap be z-scored based within regions
        zscore_regions: boolean, should heatmap be z-scored based within features
        thresh: value to threshold heatmap. Only values above this value are kept
        subset: what regions should be plotted? default is all
        compare: take a full subset?
        region: if None makes map for average and all regions individuals. 
        Otherwise 0 means average and each region is own number

    Outputs:
        Outputs a .png file for the overall heatmap and for each region. If z-scored on thresholded,
        will denote in file name using z0 (regions), z1 (features), and/or t followed by threshold.
    """

    from ..plotting import heat_map

    if subset is None:
        subset = range(0, clf.mask_num)

    if compare is True:
        overall_fi = clf.feature_importances[subset][:, subset]
    else:
        overall_fi = clf.feature_importances[:, subset]

    if np.array(subset).max() > clf.mask_num:
        print "Warning: you entered an incorrect mask index!"

    label = ""

    fi = overall_fi.mean(axis=0).T

    if zscore_regions:
        fi = np.apply_along_axis(stats.zscore, 0, fi)
        label = "z0_"
    if zscore_features:
        fi = np.apply_along_axis(stats.zscore, 1, fi)
        label = label + "z1_"

    if thresh is not None:
        fi = np.ma.masked_array(fi)
        fi.mask = fi < thresh
        label = label + "zt" + str(thresh) + "_"

    if basename is None:
        file_name = None
    else:
        base_file_name = basename + "imps_hm_" + label

    if basename is not None:
        file_name = base_file_name + "overall.png"

    if region is (None or 0):
        heat_map(fi, np.array(subset) + 1, clf.feature_names, file_name)

    for i in subset:

        if region is None or region == i + 1:

            fi = overall_fi[subset.index(i)].T

            if zscore_regions:
                fi = np.ma.masked_invalid(stats.zscore(fi, axis=0))
            if zscore_features:
                fi = stats.zscore(fi, axis=1)

            if thresh is not None:
                fi.mask = fi < thresh

            if basename is not None:
                file_name = base_file_name + str(i) + ".png"

            heat_map(fi, np.array(subset) + 1, clf.feature_names, file_name)

def average_ns(clf, result = None):
    """ For a Pairwise clf returns the n for each class (only for half of the matrix).
    Returns: (region1, region2) """
    ns = tools.mask_diagonal(np.ma.masked_array(np.empty(clf.comp_dims, np.float32)))

    for n1, row in enumerate(clf.c_data):
        for n2, data in enumerate(row):
            if n1 < n2:
                bc = np.bincount(data[1])
                if result is 'mean':
                    bc = np.mean(bc)
                elif result is 'ratio':
                    bc = bc[0] / (bc[1] * 1.0)
                ns[(n1, n2)] = bc

    return tools.copy_along_diagonal(ns)

