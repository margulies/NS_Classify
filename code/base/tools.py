#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys


class Logger():

    def __init__(self, logfile):
        self.stdout = sys.stdout
        self.log = open(logfile, 'w')
        self.old = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.log.write(text)
        self.log.flush()

    def end(self):
        self.log.close()
        sys.stdout = self.old

    def flush(self):
        self.stdout.flush()
        self.log.flush()

    def show(self, text):
        self.stdout.write(text)
        self.stdout.flush()


class ProgressBar():

    def __init__(self, total):
        self.total = total
        self.current = 0.0

    def update_progress(self, progress):
        display = '\r[{0}] {1}%'.format('#' * (progress / 10), progress)
        if sys.stdout.__class__.__name__ == 'Logging':
            sys.stdout.show(display)
        else:
            sys.stdout.write(display)
            sys.stdout.flush()

    def next(self):
        self.update_progress(int((self.current) / self.total * 100))

        if self.current == self.total:
            self.reset()
        else:
            self.current = self.current + 1

    def reset(self):
        print ""
        self.current = 0.0


def ix_subset(dataset, subset):
    return [i for i, x in enumerate(dataset.get_feature_names()) if x in subset]



def mask_diagonal(masked_array):
    """ Given a masked array, it returns the same array with the diagonals masked"""
    import numpy as np
    i, j, k = np.meshgrid(
        *map(np.arange, masked_array.shape), indexing='ij')
    masked_array.mask = (i == j)

    return masked_array

def get_index_path(pair):
    """ returns index, pair path """
    return (pair[0][1], pair[1][1]), [pair[0][0], pair[1][0]]

def invert_y(y):
    import numpy as np
    y = y + 1
    y[y == 2] = 0

    return np.flipud(y)

def calculate_feature_corr(clf):
    import numpy as np
    from scipy import stats
    f_corr = np.empty(clf.feature_importances.shape)

    for i in range(0, clf.c_data.shape[0]):
        for j in range(0, clf.c_data.shape[1]):

            if i == j:
                f_corr[i, j] = None
            else:
                data, classes = clf.c_data[i, j]

                f_corr[i, j] = np.apply_along_axis(
                    lambda x: stats.pearsonr(x, classes), 0, data)[0]

    clf.feature_corr = np.ma.masked_array(f_corr, mask=np.isnan(f_corr))
