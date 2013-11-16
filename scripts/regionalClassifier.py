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

import matplotlib.pyplot as plt

from scipy import stats


def shannons(x):
    """ Returns Shannon's Diversity Index for an np.array """
    if np.isnan(x.mean()) or x.mean() == 0.0:
        return 0.0
    else:
        x = x.astype('float')
        x = (x / x.sum())
        x = x*np.log(x)
        x = np.ma.masked_array(x, np.isnan(x))
        return ((x).sum())*-1


def heat_map(data, x_labels, y_labels, file_name=None, add_diagonal=False):

    if add_diagonal:
        new_data = []
        for i in range(0, data.shape[1]):
            x = list(data[:,i])
            x.insert(i, 0)
            new_data.append(x)

        data = np.array(new_data)

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.YlOrRd, alpha=0.8)


    fig = plt.gcf()

    fig.set_size_inches(8,11)

    # turn off the frame
    ax.set_frame_on(False)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)

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

class MaskClassifier:

    def __init__(self, dataset, masks, classifier=GradientBoostingClassifier(), 
        thresh=0.08, param_grid=None, cv=None):

        self.masklist = zip(masks, range(0, len(masks)))

        self.mask_names = [os.path.basename(os.path.splitext(os.path.splitext(mask)[0])[0]) for mask in masks]

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

    def classify(self, features=None, scoring='accuracy', dummy = True):

        iters = list(itertools.combinations(self.masklist, 2))
        prog = 0.0
        total = len(list(iters))

        self.update_progress(0)

        if features:
            self.feature_names = self.dataset.get_feature_names(features)
        else:
            self.feature_names = self.dataset.get_feature_names()


        self.feature_importances = np.ma.masked_array(np.zeros((self.mask_num,
            self.mask_num, len(self.feature_names))))

        i, j, k = np.meshgrid(*map(np.arange, self.feature_importances.shape), indexing='ij')

        self.feature_importances.mask = (i == j)

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
                    features=features, output='summary_clf', scoring=scoring)

                self.class_score[index] = output['score']

                self.fit_clfs[index] = output['clf'].fit(*self.c_data[index])

                # import ipdb; ipdb.set_trace()

                if self.param_grid: # Just get them if you used a grid

                    if isinstance(self.fit_clfs[index].estimator, LinearSVC):
                        self.feature_importances[index] = self.fit_clfs[index].fit(*self.c_data[index]).best_estimator_.coef_[0]
                    else:
                        try:
                            self.feature_importances[index] = self.fit_clfs[index].feature_importances_
                        except AttributeError:
                            pass
                elif isinstance(self.classifier, LinearSVC):
                    self.feature_importances[index] = self.fit_clfs[index].coef_[0]
                else:
                    try:
                        self.feature_importances[index] = self.fit_clfs[index].feature_importances_
                    except AttributeError:
                        pass


            self.dummy_score[index] = classify.classify_regions(self.dataset, names,
                method='Dummy' , threshold=self.thresh)['score']

            prog = prog + 1
            self.update_progress(int(prog / total * 100))

        self.class_score = np.ma.masked_array(self.class_score,
            self.class_score == 0)
        self.dummy_score = np.ma.masked_array(self.dummy_score,
            self.dummy_score == 0)

        if dummy:
            self.final_score = self.class_score - self.dummy_score
        else:
            self.final_score = self.class_score



        # Make results fill in across diagonal

        for j in range(0, self.mask_num):
            for b in range(0, self.mask_num):
                if self.final_score.mask[j, b] and not j == b:
                    self.final_score[j, b] = self.final_score[b, j]
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

        return [self.final_score[k].mean() for k in range(0,
            self.final_score.shape[0])]

    def make_mask_map(self, out_file, data):

        import tempfile
        folder = tempfile.mkdtemp()

        data = list(data)

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

        fsl.ImageMaths(in_file=folder + '/ongoing.nii.gz', op_string=' -sub 1' + ' -thr 0', out_file=out_file).run()

        print "Made" + out_file

    def get_importances(self, index, sort=True, relative=True, absolute=False, ranking=False, demeaned=False, zscore=False):
        """ get the importances with feature names given a tuple mask index
        Args:
            index: Can be an tuple index comparing two masks (2, 3),
                an integer index (i.e. average for mask 2),
                or None, which indicates overall average 
            sort: Boolean indicating if output should be sorted by importances
            relative: Should importances by normalized by highest importances
            demeaned: Demeans with respect to mean importance for entire brain

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

        if index is not None: # If None, take it out and get all
            fi = fi[index]
       

        if not isinstance(index, tuple): # If not a tuple (i.e. integer or None), get mean

            if index is None:
                fi = fi.mean(axis=0).mean(axis=0)
            else:
                fi = fi.mean(axis=0)

        if absolute:
            fi = np.abs(fi)

        if demeaned:
            [fi_all, names] = zip(*self.get_importances(None, sort=False, relative=False, absolute=absolute, ranking=ranking))
            fi = fi - np.array(fi_all)


        if relative:
            fi = 100.0 * (fi / fi.max())

        if zscore:
            fi = stats.zscore(fi)

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

        display = '\r[{0}] {1}%'.format('#' * (progress / 10), progress)
        if sys.stdout.__class__.__name__ == 'Logging':
            sys.stdout.show(display)
        else:
            sys.stdout.write(display)
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
            self.plot_importances(i-1, file_name=basename+"_imps_"+str(i)+".png", thresh=thresh)
            self.plot_importances(None, file_name=basename+"_imps_overall.png", thresh=thresh)

    def importance_stats(self, method='var', axis=0, average=True):
        """ Returns various statics on the importances for each masks
        These funcions are intended to be used to summarize how consistent or correlated 
        the importance matrices are within each region 

        axis = 0 applies across regions
        shape is len(features)
        axis = 1 is equivalent to applying to within regions

        average: average results within axis of interest?
        """

        results = []

        if axis == 0:
            rev_axis = 1
        else:
            rev_axis = 0

        for i in range(0, self.mask_num):
            region_data = np.array(filter(None, self.feature_importances[i]))

            if method == 'var':
                results.append(np.apply_along_axis(np.var, axis, region_data))

            elif method == 'cor':

                x = np.corrcoef(region_data, rowvar=rev_axis).flatten()
                results.append(np.ma.masked_array(x, np.equal(x, 1)))

            elif method == 'shannons':
                results.append(np.apply_along_axis(shannons, axis, region_data))

        if average:
            return np.array(results).mean(axis=1)
        else:
            return np.array(results)

    def accuracy_stats(self, method='shannons'):
        results = []
        for row in range(0, self.mask_num):
            if method == 'shannons':
                results.append(shannons(self.final_score[row]))
            if method == 'var':
                results.append(self.final_score[row].var())

        return results

    def region_heatmap(self, basename=None, zscore=False):

        fi = self.feature_importances.mean(axis=0).T

        if zscore:
            fi = np.apply_along_aix(stats.zscore, 0, fi)

        heat_map(self.feature_importances.mean(axis=0).T, range(0,self.mask_num), self.feature_names, basename + "imps_hm_overall.png")
        for i in range(0, self.mask_num):

            fi = self.feature_importances[i].T

            if basename is None:
                file_name = None
            else:
                file_name = basename + "imps_hm_" + str(i) + ".png"

            heat_map(fi, range(0,self.mask_num), self.feature_names, file_name)

    def save(self, filename, keep_dataset=False):
        if not keep_dataset:
            self.datset= []
        import cPickle
        cPickle.dump(self, open(filename, 'wb'), -1)




    # def features_to_topics(self, topic_weights, importances):
    #     topic_imps = np.array([topic_weights_feature(topic_weights, pair[1])*pair[0] for pair in importances]).mean(axis=0)

    #     ## Then just zip back up to topic numbers





