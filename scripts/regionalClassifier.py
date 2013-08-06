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


def shannons(x):
    """ Returns Shannon's Diversity Index for an np.array """
    x.astype('float')
    x = (x / x.sum())

    return ((x*np.log(x)).sum())*-1





class MaskClassifier:

    def __init__(self, dataset, masks, classifier=GradientBoostingClassifier(), 
        thresh=0.08, param_grid=None, cv=None):

        self.masklist = zip(masks, range(0, len(masks)))

        self.mask_num = len(self.masklist)

        # Where to store differences, results, n's and dummy results

        self.class_score = np.zeros((self.mask_num, self.mask_num))
        self.dummy_score = np.zeros((self.mask_num, self.mask_num))

        self.classifier = classifier
        self.dataset = dataset
        self.thresh = thresh
        self.param_grid = param_grid

        self.fit_clfs = np.empty((self.mask_num, self.mask_num), object)  # Fitted classifier
        self.c_data = np.empty((self.mask_num, self.mask_num), tuple)  # Actual data

        self.cv = cv

        self.status = 0

        if isinstance(self.classifier, RFE):
            self.feature_ranking = np.empty((self.mask_num, self.mask_num), object)  # Fitted classifier
        else:
            self.feature_ranking = None

    def classify(self, features=None):

        iters = list(itertools.combinations(self.masklist, 2))
        prog = 0.0
        total = len(list(iters))

        self.update_progress(0)

        if features:
            self.feature_names = self.dataset.get_feature_names(features)
        else:
            self.feature_names = self.dataset.get_feature_names()

        self.feature_importances = np.empty((self.mask_num,
            self.mask_num), object)

        for pairs in iters:

            index = (pairs[0][1], pairs[1][1]) # Tuple numeric index of pairs
            names = [pairs[0][0], pairs[1][0]] # Actual paths to masks

            self.c_data[index] = classify.get_studies_by_regions(self.dataset, 
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
                    self.fit_clfs[index].fit(*self.c_data[index]).best_estimator_.feature_importances_
                elif isinstance(self.classifier, GradientBoostingClassifier): # Refit if not param_grid

                    self.feature_importances[index] = self.fit_clfs[index].feature_importances_

                elif isinstance(self.classifier, LinearSVC):
                    self.feature_importances[index] = self.fit_clfs[index].coef_[0]


            self.dummy_score[index] = classify.classify_regions(self.dataset, names,
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
                    if isinstance(self.classifier, LinearSVC):
                        self.feature_importances[j, b] = self.feature_importances[b, j] * -1
                    else:
                        self.feature_importances[j, b] = self.feature_importances[b, j]
                    
                    if self.feature_ranking is not None:
                        self.feature_ranking[j, b] = self.feature_ranking[b, j]

        self.status = 1

    def get_mask_averages(self):
        return [self.diffs[k].mean() for k in range(0,
                self.diffs.shape[0])]

    def get_mask_diversity(self):

        mask_shannons = []

        for i in range(0, self.mask_num):
            imps = []
            for j in range(0, self.mask_num):
                if i is not j:
                    imp, name =  zip(*self.get_importances([i,j], relative=False, sort=False))

                    imps.append(imp)

            imps = np.array(imps)

            s = []

            for row in range(0, imps.shape[0]):
                s.append(shannons(imps[row, :]))

            mask_shannons.append(np.array(s).mean())


        return mask_shannons

       

    def make_mask_map(self, out_file, data):

        import tempfile
        folder = tempfile.mkdtemp()

        (masks, num) = zip(*self.masklist)

        for (n, v) in enumerate(data):

            fsl.ImageMaths(in_file=masks[n], op_string=' -add '
                           + str(v) + ' -thr ' + str(v + 0.001),
                           out_file=folder + '/' + str(n) + '.nii.gz'
                           ).run()

        fsl.ImageMaths(in_file=folder + '/0.nii.gz', op_string=' -add '+ folder + '/1', out_file=folder + '/ongoing.nii.gz').run()

        if self.mask_num > 1:         
            for n in range(2, self.mask_num):
                fsl.ImageMaths(in_file=folder + '/ongoing.nii.gz', op_string=' -add '
                           + folder + '/' + str(n), out_file=folder + '/ongoing.nii.gz').run()

        fsl.ImageMaths(in_file=folder + '/ongoing.nii.gz', op_string=' -sub 1' + ' -thr 0', out_file=out_file)

        print "Made" + out_file

    def get_importances(self, index, sort=True, relative=True, absolute=False, ranking=False, demeaned=False):
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


        if not self.status:
            raise Exception("You haven't finished classification yet!")

        if ranking:
            if not isinstance(self.classifier, RFE):
                raise Exception("Hey! You didn't even set an RFE classifier!")
            fi = self.feature_ranking
        else:
            fi = self.feature_importances

        if index: # If None, take it out and get all
            fi = fi[index]
       

        if not isinstance(index, tuple): # If not a tuple (i.e. integer or None), get mean
            fi = np.ma.masked_array(fi, np.equal(fi, None))

            if index is None:
                fi = np.array([fi[col].mean() for col in range(0, fi.shape[0])]).mean(axis=0)
            else:
                fi = np.array(fi.mean())

        if absolute:
            fi = np.abs(fi)

        if demeaned:
            [fi_all, names] = zip(*self.get_importances(None, sort=False, relative=False, absolute=absolute, ranking=ranking))
            fi = fi - np.array(fi_all)



        if relative:
            fi = 100.0 * (fi / fi.max())

        imps = [(i, self.feature_names[num]) for (num, i) in
                enumerate(fi)]

        if sort:
            from operator import itemgetter
            imps.sort(key=itemgetter(0))

        return imps

    def min_features(self):
        return np.array([n_features for n_features in self.fit_clfs.flatten()]).mean()

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

        if sys.stdout.__class__.__name__ == 'Logging':
            sys.stdout.show('\r[{0}] {1}%'.format('#' * (progress / 10),
                         progress))
        else:
            sys.stdout.write('\r[{0}] {1}%'.format('#' * (progress / 10),
                             progress))
            sys.stdout.flush()

        if progress == 100:
            print

    def get_best_features(self, n, ranking=True):
        """ Gets the n best features across all comparisons from a RFE classifier """

        return self.get_importances(None, absolute=True, ranking=ranking)[-n:]

    def topic_weights_feature(self, topic_weights, feature):
        """ Returns topic weights for a feature in order of topic_weights
        Best if that order is sorted """

        return topic_weights[np.where(topic_weights == feature )[0]][:, 2]

    def save_region_importance_plots(self, basename, thresh=20):
        for i in range(1, self.mask_num):
            self.plot_importances(i-1, file_name=basename+"_"+str(i)+".png", thresh=thresh)
            self.plot_importances(None, file_name=basename+"_overall.png", thresh=thresh)

    # def features_to_topics(self, topic_weights, importances):
    #     topic_imps = np.array([topic_weights_feature(topic_weights, pair[1])*pair[0] for pair in importances]).mean(axis=0)

    #     ## Then just zip back up to topic numbers




