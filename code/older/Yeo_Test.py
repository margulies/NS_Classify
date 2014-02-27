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


dataset = Dataset.load('../data/pickled.pkl')

masklist = ['7Networks_Liberal_1.nii.gz', '7Networks_Liberal_2.nii.gz',
            '7Networks_Liberal_3.nii.gz', '7Networks_Liberal_4.nii.gz',
            '7Networks_Liberal_5.nii.gz', '7Networks_Liberal_6.nii.gz',
            '7Networks_Liberal_7.nii.gz']

rootdir = '../masks/Yeo_JNeurophysiol11_MNI152/standardized/'


class maskClassifier:
    def __init__(self, classifier=GradientBoostingClassifier(), param_grid={'max_features': np.arange(2, 140, 44), 'n_estimators': np.arange(5, 141, 50),
          'learning_rate': np.arange(0.05, 1, 0.1)}, thresh = 0.08)


diffs = {}

results = np.zeros((7, 7))

ns = np.zeros((7, 7))

resultsDummy = np.zeros((7, 7))
iters = list(itertools.combinations(masklist, 2))

param_grid = 

fitClfs = np.empty((7, 7), object)

c_data = np.empty((7, 7), tuple)

prog = 0.0
total = len(list(iters))
update_progress(0)

for pairs in iters:

    output = classify.classify_regions(dataset, [rootdir + pairs[0],
            rootdir + pairs[1]],
            classifier=GradientBoostingClassifier(), param_grid=param_grid,
            threshold=thresh, output='summary_clf')

    results[int(re.findall('[0-9]', pairs[0])[1]) - 1,
            int(re.findall('[0-9]', pairs[1])[1]) - 1] = output['score']

    ns[int(re.findall('[0-9]', pairs[0])[1]) - 1, int(re.findall('[0-9]'
       , pairs[1])[1]) - 1] = output['n'][0] + output['n'][1]

    fitClfs[int(re.findall('[0-9]', pairs[0])[1]) - 1,
            int(re.findall('[0-9]', pairs[1])[1]) - 1] = output['clf']

    c_data[int(re.findall('[0-9]', pairs[0])[1]) - 1,
           int(re.findall('[0-9]', pairs[1])[1]) - 1] = \
        classify.get_studies_by_regions(dataset, [rootdir + pairs[0],
            rootdir + pairs[1]], threshold=thresh)

    dummyoutput = classify.classify_regions(dataset, [rootdir
            + pairs[0], rootdir + pairs[1]],
            method='Dummy', threshold=thresh)

    resultsDummy[int(re.findall('[0-9]', pairs[0])[1]) - 1,
                 int(re.findall('[0-9]', pairs[1])[1]) - 1] = \
        dummyoutput['score']
    prog = prog + 1
    update_progress(int(prog / total * 100))

results = np.ma.masked_array(results, results == 0)
resultsDummy = np.ma.masked_array(resultsDummy, resultsDummy == 0)
diffs = results - resultsDummy

for j in range(0, 7):
    for b in range(0, 7):
        if diffs.mask[j, b]:
            diffs[j, b] = diffs[b, j]
            fitClfs[j, b] = fitClfs[b, j]
            ns[j, b] = ns[b, j]
            c_data[j, b] = ns[b, j]

maskaverage = [diffs[k].mean() for k in range(0, diffs.shape[0])]

make_average_map(maskaverage)

def make_average_map(maskaverage, outfile = ../results/Yeo_7Networks_AvgClass.nii.gz)
	import tempfile
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
	               + ' -thr 0', out_file).run()


def get_importances(index, fitClfs, sort=True, relative=True):
    """ get the importances with feature names given a tuple mask index """

    fi = fitClfs[index].clf.best_estimator_.feature_importances_
    if relative:
    	fi = 100.0 * (fi / fi.max())
    fn = dataset.get_feature_names()
    imps = [(i, fn[num]) for (num, i) in enumerate(fi)]

    if sort:
        from operator import itemgetter
        imps.sort(key=itemgetter(0))



    return imps

def plot_importances(index, fitClfs, thresh=0.01):
	""" Plot importances for a given index """
	import pylab as pl

	[imps, names] = zip(*get_importances(index, fitClfs))

	imps = np.array(imps)
	imps = imps[imps>0.01]

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

def update_progress(progress):
    sys.stdout.write('\r[{0}] {1}%'.format('#' * (progress / 10),
                     progress))