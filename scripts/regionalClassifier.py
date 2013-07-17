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

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.metrics import zero_one_loss


class maskClassifier:

    def __init__(self, dataset, masks,
                 classifier=GradientBoostingClassifier(), thresh=0.08,
                 param_grid={'max_features': np.arange(2, 140, 44),
                 'n_estimators': np.arange(5, 141, 50),
                 'learning_rate': np.arange(0.05, 1, 0.1)}, cv=None):

        self.masklist = zip(masks, range(0, len(masks)))

        self.mask_num = len(self.masklist)

        # Where to store differences, results, n's and dummy results

        self.diffs = {}
        self.class_score = np.zeros((self.mask_num, self.mask_num))
        self.dummy_score = np.zeros((self.mask_num, self.mask_num))

        self.classifier = classifier
        self.dataset = dataset
        self.thresh = thresh
        self.param_grid = param_grid

        self.fit_clfs = np.empty((self.mask_num, self.mask_num), object)  # Fitted classifier
        self.c_data = np.empty((self.mask_num, self.mask_num), tuple)  # Actual data

        self.cv = cv

        if isinstance(self.classifier, RFE):
             self.feature_ranking = np.empty((self.mask_num, self.mask_num), object)  # Fitted classifier

    def classify(self, features=None):


        iters = list(itertools.combinations(self.masklist, 2))
        prog = 0.0
        total = len(list(iters))

        self.update_progress(0)

        if features:
            self.features = features
        else:
            self.features = self.dataset.get_feature_names()

        self.feature_importances = np.empty((self.mask_num,
                    self.mask_num), object)

        for pairs in iters:

            index = (pairs[0][1], pairs[1][1])
            names = [pairs[0][0], pairs[1][0]]

            self.c_data[index] = classify.get_studies_by_regions(dataset, 
                        names, threshold=self.thresh, features=features, regularization='scale')

            if isinstance(self.classifier, RFE):

                self.classifier.fit(*self.c_data[index])

                self.fit_clfs[index] = self.classifier

                self.class_score[index] = self.classifier.score(*self.c_data[index])

                self.feature_importances[index] = self.classifier.estimator_.coef_[0]

                self.feature_ranking[index] = self.classifier.ranking_

            else:
                output = classify.classify_regions(self.dataset, names,
                        classifier=self.classifier,
                        param_grid=self.param_grid, threshold=self.thresh,
                        features=features, output='summary_clf')

                self.class_score[index] = output['score']

                self.fit_clfs[index] = output['clf'].fit(*self.c_data[index])

                if self.param_grid: # Just get them if you used a grid
                    self.feature_importances[index] = \
                        self.fit_clfs[index].fit(*self.c_data[index]).feature_importances_
                elif isinstance(self.classifier, GradientBoostingClassifier): # Refit if not param_grid
                    self.feature_importances[index] = self.fit_clfs[index].feature_importances_
                elif isinstance(self.classifier, LinearSVC):
                    self.feature_importances[index] = self.fit_clfs[index].coef_[0]


            self.dummy_score[index] = classify.classify_regions(dataset, names,
                method='Dummy' , threshold=self.thresh)['score']

            prog = prog + 1
            self.update_progress(int(prog / total * 100))

        self.class_score = np.ma.masked_array(self.class_score,
                self.class_score == 0)
        self.dummy_score = np.ma.masked_array(self.dummy_score,
                self.dummy_score == 0)

        self.diffs = self.class_score - self.dummy_score


        # Make results fill in across diagonal

        for j in range(0, self.mask_num):
            for b in range(0, self.mask_num):
                if self.diffs.mask[j, b] and not j == b:
                    self.diffs[j, b] = self.diffs[b, j]
                    self.fit_clfs[j, b] = self.fit_clfs[b, j]
                    self.c_data[j, b] = self.c_data[b, j]
                    self.feature_importances[j, b] = self.feature_importances[b, j] * -1

        self.feature_names = self.dataset.get_feature_names(features)

    def get_mask_averages(self):
        return [self.diffs[k].mean() for k in range(0,
                self.diffs.shape[0])]

    def make_average_map(self,
                         out_file='../results/Yeo_7Networks_AvgClass.nii.gz'):

        import tempfile

        maskaverage = self.get_mask_averages()

        folder = tempfile.mkdtemp()
        for (n, v) in enumerate(maskaverage):

            fsl.ImageMaths(in_file=rootdir + '7Networks_Liberal_'
                           + str(n + 1) + '.nii.gz', op_string=' -add '
                           + str(v) + ' -thr ' + str(v + 0.001),
                           out_file=folder + '/' + str(n) + '.nii.gz'
                           ).run()

        fsl.ImageMaths(in_file=folder + '/0.nii.gz', op_string=' -add '
                       + folder + '/1' + ' -add ' + folder + '/2'
                       + ' -add ' + folder + '/3' + ' -add ' + folder
                       + '/4' + ' -add ' + folder + '/5' + ' -add '
                       + folder + '/6' + ' -sub 1' + ' -thr 0',
                       out_file=out_file).run()


    def get_importances(self, index, sort=True, relative=True, absolute=False, ranking=False):
        """ get the importances with feature names given a tuple mask index
        Args:
            index: Can be an tuple index comparing two masks (2, 3),
                an integer index (i.e. average for mask 2),
                or None, which indicates overall average 

            sort: Boolean indicating if output should be sorted by importances
            relative: Should importances by normalized by highest importances

        Output:
            A list of tuples with importance feature pairs
        """

        if ranking:
            fi = self.feature_ranking
        else:
            fi = self.feature_importances

        if not index: # If None, take it out and get all
            fi = fi
        else:
            fi = fi[index]
       

        if not isinstance(index, tuple): # If not a tuple (i.e. integer or None), get mean
            fi = np.array(np.ma.masked_array(fi, np.equal(fi, None)).mean())

        if absolute:
            fi = np.abs(fi)

        if relative:
            fi = 100.0 * (fi / fi.max())

        imps = [(i, self.feature_names[num]) for (num, i) in
                enumerate(fi)]

        if sort:
            from operator import itemgetter
            imps.sort(key=itemgetter(0))

        return imps

    def min_features(self):
        n_features = []
        for d1 in self.fit_clfs:
            for d2 in d1:
                if d2:
                    n_features.append(d2.n_features_)

        return np.array(n_features).mean()



    def plot_importances(self, index, thresh=20, file_name=None, absolute=False, ranking=False):
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

        [imps, names] = zip(*self.get_importances(index, absolute=absolute, ranking=ranking))

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

    def update_progress(self, progress):
        sys.stdout.write('\r[{0}] {1}%'.format('#' * (progress / 10),
                         progress))
        sys.stdout.flush()


dataset = Dataset.load('../data/dataset.pkl')

masks = ['7Networks_Liberal_1.nii.gz', '7Networks_Liberal_2.nii.gz',
         '7Networks_Liberal_3.nii.gz', '7Networks_Liberal_4.nii.gz',
         '7Networks_Liberal_5.nii.gz', '7Networks_Liberal_6.nii.gz',
         '7Networks_Liberal_7.nii.gz']

rootdir = '../masks/Yeo_JNeurophysiol11_MNI152/standardized/'

masklist = [rootdir + m for m in masks]

import csv
readfile = open('../data/reduced_features.csv', 'rbU')
wr = csv.reader(readfile, quoting=False)
reduced_features = [word[0] for word in wr]
reduced_features = [word[2:-1] for word in reduced_features]

# yeoClass = maskClassifier(dataset, masklist, param_grid=None,
#                           classifier=GradientBoostingClassifier(learning_rate=0.25,
#                           max_features=50, n_estimators=1000))

# yeoClass.classify(features=reduced_features, calculate_importances=True)


# for i in range(1, 7):
#     yeoClass.plot_importances(i-1, file_name="../results/Yeo_imps_"+str(i)+".png")
# yeoClass.plot_importances(None, file_name="../results/Yeo_imps_overall.png")