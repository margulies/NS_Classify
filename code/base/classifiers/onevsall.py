#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats

def region_heatmap(clf, basename=None, zscore_regions=False, zscore_features=False, thresh=None, subset=None):
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

    fi = clf.feature_importances[subset].T

    if np.array(subset).max() > clf.mask_num:
        print "Warning: you entered an incorrect mask index!"

    label = ""

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
        file_name = basename + "imps_hm_" + label + ".png"

    heat_map(fi, np.array(subset) + 1, clf.feature_names, file_name)

def average_ns(clf, mean=True):
	""" For a OvA clf returns the n for each class.
	Returns: (rest_of_brain_n, region_n) """
	region_n = []
	others_n = []
	for data in clf.c_data:
	    bc = np.bincount(data[1])
	    region_n.append(bc[0])
	    others_n.append(bc[1])
	if mean is True:
		result = (int(np.array(others_n).mean()), int(np.array(region_n).mean()))
	else:
		result = zip(others_n, region_n)

	return result