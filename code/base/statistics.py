#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

def shannons(x):
    """ Returns Shannon's Diversity Index for an np.array """
    if np.isnan(x.mean()) or x.mean() == 0.0:
        return 0.0
    else:
        x = x.astype('float')
        x = (x / x.sum())
        x = x * np.log(x)
        x = np.ma.masked_array(x, np.isnan(x))
        return ((x).sum()) * -1
