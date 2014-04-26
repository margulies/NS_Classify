#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

def shannons(x):
    """ Returns Shannon's Diversity Index for an np.array """
    if np.isnan(x.mean()) or x.mean() == 0.0:
        return 0.0
    else:
    	# Don't allow values below 0
    	x = np.abs(x.min()) + x + 0.00001
        x = x.astype('float')
        x = (x / x.sum())
        x = x * np.log(x)
        x = np.ma.masked_array(x, np.isnan(x))
        return ((x).sum()) * -1

def get_roc(x, y):
	from sklearn.metrics import roc_curve, auc
	fpr, tpr, thresholds = roc_curve(x, y)
	return auc(fpr, tpr)