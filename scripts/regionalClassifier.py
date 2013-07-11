#!/usr/bin/python
# -*- coding: utf-8 -*-

# Here I use Yeo to test Neurosynth's classify functions
from neurosynth.base.dataset import Dataset
from neurosynth.analysis import classify
import os
import itertools
import re
import numpy as np
import pdb
import sys
from nipype.interfaces import fsl
from sklearn.ensemble import GradientBoostingClassifier


class maskClassifier:
    def __init__(self, dataset, classifier=GradientBoostingClassifier(), thresh = 0.08, 
        param_grid={'max_features': np.arange(2, 140, 44), 'n_estimators': np.arange(5, 141, 50),
          'learning_rate': np.arange(0.05, 1, 0.1)}):

        # Where to store differences, results, n's and dummy results
        self.diffs = {}
        self.class_score = np.zeros((7, 7))
        self.ns = np.zeros((7, 7))
        self.dummy_score = np.zeros((7, 7))

        self.classifier = classifier
        self.dataset = dataset
        self.thresh = thresh
        self.param_grid = param_grid

        self.fit_clfs = np.empty((7, 7), object) # Fitted classifier
        self.c_data = np.empty((7, 7), tuple)    # Actual data

    def classify(self, masks):

        masklist = zip(masks, range(0, len(masks)))

        iters = list(itertools.combinations(masklist, 2))
        prog = 0.0
        total = len(list(iters))
        self.update_progress(0)


        for pairs in iters:

            output = classify.classify_regions(self.dataset, [pairs[0][0], pairs[1][0]],
                    classifier=self.classifier, param_grid=self.param_grid,
                    threshold=self.thresh, output='summary_clf')


            self.class_score[pairs[0][1], pairs[1][1]] = output['score']

            self.ns[pairs[0][1], pairs[1][1]] = output['n'][0] + output['n'][1]

            self.fit_clfs[pairs[0][1], pairs[1][1]] = output['clf']

            self.c_data[pairs[0][1], pairs[1][1]] = classify.get_studies_by_regions(dataset, [pairs[0][0],
                    pairs[1][0]], threshold=self.thresh)

            self.dummy_score[pairs[0][1], pairs[1][1]] = classify.classify_regions(dataset, 
                [pairs[0][0], pairs[1][0]], method='Dummy', threshold=self.thresh)['score']

            prog = prog + 1
            self.update_progress(int(prog / total * 100))

        self.class_score = np.ma.masked_array(self.class_score, self.class_score == 0)
        self.dummy_score = np.ma.masked_array(self.dummy_score, self.dummy_score == 0)

        self.diffs = self.class_score - self.dummy_score

        # Make results fill in across diagonal
        for j in range(0, 7):
            for b in range(0, 7):
                if self.diffs.mask[j, b]:
                    self.diffs[j, b] = self.diffs[b, j]
                    self.fit_clfs[j, b] = self.fit_clfs[b, j]
                    self.ns[j, b] = self.ns[b, j]
                    self.c_data[j, b] = self.c_data[b, j]

    def get_mask_averages(self):
        return [self.diffs[k].mean() for k in range(0, self.diffs.shape[0])]


    def make_average_map(self, out_file = '../results/Yeo_7Networks_AvgClass.nii.gz'):

        import tempfile

        maskaverage = self.get_mask_averages()

    	folder = tempfile.mkdtemp()
    	for (n, v) in enumerate(maskaverage):
    	    fsl.ImageMaths(in_file=rootdir + '7Networks_Liberal_' + str(n + 1)
    	                   + '.nii.gz', op_string=' -add ' + str(v) + ' -thr '
    	                   + str(v + 0.001), out_file=folder + '/' + str(n)
    	                   + '.nii.gz').run()

    	fsl.ImageMaths(in_file=folder + '/0.nii.gz', op_string=' -add '
    	               + folder + '/1' + ' -add ' + folder + '/2' + ' -add '
    	               + folder + '/3' + ' -add ' + folder + '/4' + ' -add '
    	               + folder + '/5' + ' -add ' + folder + '/6' + ' -sub 1'
    	               + ' -thr 0', out_file=out_file).run()


    def get_importances(self, index, sort=True, relative=True):
        """ get the importances with feature names given a tuple mask index """

        if self.param_grid:
            fi = self.fit_clfs[index].clf.best_estimator_.feature_importances_
        else:
            fi = self.fit_clfs[index].clf.fit(self.c_data[index][0], yeoClass.c_data[index][1]).feature_importances_

        if relative:
        	fi = 100.0 * (fi / fi.max())
        fn = dataset.get_feature_names()
        imps = [(i, fn[num]) for (num, i) in enumerate(fi)]

        if sort:
            from operator import itemgetter
            imps.sort(key=itemgetter(0))

        return imps

    def plot_importances(self, index, thresh=0.01):
    	""" Plot importances for a given index """
    	import pylab as pl

    	[imps, names] = zip(*self.get_importances(index))

    	imps = np.array(imps)
    	imps = imps[imps>thresh]

    	names = np.array(names)
    	names[:len(imps)]

    	sorted_idx = np.argsort(imps)
    	pos = np.arange(sorted_idx.shape[0]) + .5
    	pl.subplot(1, 2, 2)

    	pl.barh(pos, imps[sorted_idx], align='center')
    	pl.yticks(pos, names[sorted_idx])
    	pl.xlabel('Relative Importance')
    	pl.title('Variable Importance')
    	pl.show()

    def update_progress(self, progress):
        sys.stdout.write('\r[{0}] {1}%'.format('#' * (progress / 10),
                         progress))

dataset = Dataset.load('../data/pickled.pkl')

masks = ['7Networks_Liberal_1.nii.gz', '7Networks_Liberal_2.nii.gz',
            '7Networks_Liberal_3.nii.gz', '7Networks_Liberal_4.nii.gz',
            '7Networks_Liberal_5.nii.gz', '7Networks_Liberal_6.nii.gz',
            '7Networks_Liberal_7.nii.gz']

rootdir = '../masks/Yeo_JNeurophysiol11_MNI152/standardized/'

masklist = [rootdir + m for m in masks]