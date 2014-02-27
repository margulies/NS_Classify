#!/usr/bin/python
# -*- coding: utf-8 -*-

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
from sklearn.preprocessing import binarize

def plot_min_max_fi(clf):
    density_plot(clf.feature_importances[tuple(np.where(sh_1 == sh_1.min())[0])], file_name="../results/diagonstic/sh_1_min.png")
    density_plot(clf.feature_importances[tuple(np.where(sh_1 == sh_1.max())[0])], file_name="../results/diagonstic/sh_1_max.png")
    density_plot(clf.feature_importances[np.where(sh == sh.max())[0]][:, np.where(sh == sh.max())[1]], file_name="../results/diagonstic/sh_0_max.png")
    density_plot(clf.feature_importances[np.where(sh == sh.min())[0]][:, np.where(sh == sh.min())[1]], file_name="../results/diagonstic/sh_0_min.png")

def ix_subset(dataset, subset):
     return [i for i, x in enumerate(dataset.get_feature_names()) if x in subset]



def get_ns_for_pairs(a_b_c):
    """Convert `f([1,2])` to `f(1,2)` call."""
    from copy import deepcopy
    dataset, pairs, thresh = a_b_c

    index = (pairs[0][1], pairs[1][1]) # Tuple numeric index of pairs
    names = [pairs[0][0], pairs[1][0]] # Actual paths to masks

    X, y = classify.get_studies_by_regions(dataset, names, thresh)

    n = np.bincount(y)
    return (index, n)

def calculate_feature_corr(clf):
    import numpy as np
    from scipy import stats

    f_corr =  np.empty(clf.feature_importances.shape)

    for i in range(0, clf.c_data.shape[0]):
        for j in range(0, clf.c_data.shape[1]):

            if i == j:
                f_corr[i, j] = None
            else:
                data, classes = clf.c_data[i, j]

                f_corr[i, j] = np.apply_along_axis(lambda x: stats.pearsonr(x, classes), 0, data)[0]

    clf.feature_corr = np.ma.masked_array(f_corr, mask=np.isnan(f_corr))
    
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

def density_plot(data, file_name=None, covariance_factor = .2):
    """ Generate a density plot """
    data = np.array(data)
    density = stats.gaussian_kde(data)
    xs = np.linspace(0,data.max()+data.max()/10,200)
    density.covariance_factor = lambda : covariance_factor
    density._compute_covariance()
    plt.plot(xs,density(xs))

    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)

def heat_map(data, x_labels, y_labels, file_name=None):

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

    def calculate_ns(self):
        from multiprocessing import Pool

        mask_pairs = list(itertools.combinations(self.masklist, 2))
        self.ns =  np.ma.masked_array(np.empty((self.mask_num, self.mask_num, 2)), True)

        prog = 0.0
        total = len(list(mask_pairs))

        self.update_progress(0)

        p = Pool()

        for index, n in p.imap_unordered(get_ns_for_pairs, itertools.izip(itertools.repeat(self.dataset), mask_pairs, itertools.repeat(self.thresh))):
            self.ns[index] = n
            prog = prog + 1
            self.update_progress(int(prog / total * 100))


    def classify(self, features=None, scoring='accuracy', dummy = True, X_threshold=None):

        iters = list(itertools.permutations(self.masklist, 2))
        prog = 0.0
        total = len(list(iters))

        self.update_progress(0)

        if features:
            self.feature_names = features
        else:
            self.feature_names = self.dataset.get_feature_names()

        # Make feature importance grid w/ masked diagonals
        self.feature_importances = np.ma.masked_array(np.zeros((self.mask_num,
            self.mask_num, len(self.feature_names))))

        i, j, k = np.meshgrid(*map(np.arange, self.feature_importances.shape), indexing='ij')

        self.feature_importances.mask = (i == j)

        for pairs in iters:

            index = (pairs[0][1], pairs[1][1]) # Tuple numeric index of pairs
            names = [pairs[0][0], pairs[1][0]] # Actual paths to masks

            if self.c_data[index] is None:
                X, y = classify.get_studies_by_regions(self.dataset, 
                    names, threshold=self.thresh, features=features, regularization='scale')

            if X_threshold is not None:
                X = binarize(X, X_threshold)

            # if features is not None:
            #     X = X[:, classify.get_feature_order(self.dataset, self.feature_names)]

            self.c_data[index] = (X, y)

            if isinstance(self.classifier, RFE):

                self.classifier.fit(*self.c_data[index])

                self.fit_clfs[index] = self.classifier

                self.class_score[index] = self.classifier.score(*self.c_data[index])

                self.feature_importances[index] = self.classifier.estimator_.coef_[0]

                self.feature_ranking[index] = self.classifier.ranking_

            else:
                output = classify.classify(X, y, classifier = self.classifier, output = 'summary_clf', cross_val = '4-Fold',
                    class_weight = 'auto', scoring=scoring, param_grid=self.param_grid)

                self.class_score[index] = output['score']

                self.fit_clfs[index] = output['clf'].fit(*self.c_data[index])

                # import ipdb; ipdb.set_trace()

                if self.param_grid: # Just get them if you used a grid
                    try:
                        self.feature_importances[index] = self.fit_clfs[index].best_estimator_.coef_[0]
                    except AttributeError:
                        try:
                            self.feature_importances[index] = self.fit_clfs[index].feature_importances_
                        except AttributeError:
                            pass
                else:
                    try:
                        self.feature_importances[index] = self.fit_clfs[index].coef_[0]
                    except AttributeError:
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
        # for j in range(0, self.mask_num):
        #     for b in range(0, self.mask_num):
        #         if self.final_score.mask[j, b] and not j == b:
        #             self.final_score[j, b] = self.final_score[b, j]
        #             self.fit_clfs[j, b] = self.fit_clfs[b, j]
        #             self.c_data[j, b] = self.c_data[b, j]
        #             if isinstance(self.classifier, LinearSVC):
        #                 self.feature_importances[j, b] = self.feature_importances[b, j] * -1
        #             else:
        #                 self.feature_importances[j, b] = self.feature_importances[b, j]
                    
        #             if self.feature_ranking is not None:
        #                 self.feature_ranking[j, b] = self.feature_ranking[b, j]

        self.status = 1

    def get_mask_averages(self, precision=None, subset=None):

        if subset is not None:
            final_score = self.final_score[subset][:, subset]
        else:
            final_score = self.final_score

        averages = [final_score[k].mean() for k in range(0,
            final_score.shape[0])]

        if precision is not None:
            averages = [round(x, precision) for x in averages]

        return averages

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

    def save_region_importance_plots(self, basename, thresh=20):
        for i in range(1, self.mask_num):
            self.plot_importances(i-1, file_name=basename+"_imps_"+str(i)+".png", thresh=thresh)
            self.plot_importances(None, file_name=basename+"_imps_overall.png", thresh=thresh)

    def importance_stats(self, method='shannons', axis=0, average=True, subset=None):
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
            subset = range(0, self.mask_num)

        results = []

        fi = self.feature_importances[subset][:, subset]

        if axis == 0:
            rev_axis = 1
        else:
            rev_axis = 0

        for i in subset:
            region_data = np.array(filter(None, fi[subset.index(i)]))

            if method == 'var':
                results.append(np.apply_along_axis(np.var, axis, region_data))

            elif method == 'cor':
                x = np.corrcoef(region_data, rowvar=rev_axis).flatten()
                results.append(np.ma.masked_array(x, np.equal(x, 1)))

            elif method == 'shannons':
                results.append(np.apply_along_axis(shannons, axis, region_data))

        results = np.array(results)

        if axis == 1:
            results = np.ma.masked_array(results)
            i, j = np.meshgrid(*map(np.arange, results.shape), indexing='ij')
            results.mask = (i == j)

        if average:
            return results.mean(axis=1)
        else:
            return results

    def accuracy_stats(self, method='shannons', subset=None):

        if subset is None:
            subset = range(0, self.mask_num)

        fs = self.final_score[subset][:, subset]

        results = []
        for row in subset:
            if method == 'shannons':
                results.append(shannons(fs[subset.index(row)]))
            elif method == 'var':
                results.append(fs[subset.index(row)].var())

        return results

    def minN_by_region(self):
        """ Returns the average N for the smallest class in each comparison for each region """
        results = []
        for i in self.c_data:
            r = []
            for j in i:
                if j is not None:
                    r.append(np.bincount(j[1])[np.bincount(j[1]) != 0].min())

            results.append(np.array(r).mean())
        return results

    def region_heatmap(self, basename=None, zscore_regions=False, zscore_features=False, thresh=None, subset=None):
        """" Makes a heatmap of the importances of the classification. Makes an overall average heatmap
        as well as a heatmap for each individual region. Optionally, you can specify the heatmap to be
        z-scored. You can also specify a threshold.

        Args:
            basename: string, base directory and file name
            zscore_regions: boolean, should heatmap be z-scored based within regions
            zscore_regions: boolean, should heatmap be z-scored based within features
            thresh: value to threshold heatmap. Only values above this value are kept
            subset: what regions should be plotted? default is all

        Outputs:
            Outputs a .png file for the overall heatmap and for each region. If z-scored on thresholded,
            will denote in file name using z0 (regions), z1 (features), and/or t followed by threshold.
        """

        if subset is None:
            subset = range(0, self.mask_num)

        overall_fi = self.feature_importances[subset][:, subset]
        if np.array(subset).max() > self.mask_num:
            print "Warning: you entered an incorrect mask index!"

        fi = overall_fi.mean(axis=0).T

        z0 = ""
        z1 = ""
        t = ""

        if zscore_regions:
            fi = np.apply_along_axis(stats.zscore, 0, fi)
            z0 = "z0_"
        if zscore_features:
            fi = np.apply_along_axis(stats.zscore, 1, fi)
            z1 = "z1_"

        if thresh is not None:
            fi = np.ma.masked_array(fi)
            fi.mask = fi < zthresh
            t = "zt" + str(zthresh) + "_"

        heat_map(fi, np.array(subset) + 1, self.feature_names, basename + "imps_hm_" +  z0 + z1 + t + "overall.png")

        for i in subset:

            fi = overall_fi[subset.index(i)].T

            if zscore_regions:
                fi = np.ma.masked_invalid(stats.zscore(fi, axis=0))
            if zscore_features:
                fi = stats.zscore(fi, axis=1)

            if thresh is not None:
                fi.mask = fi < zthresh

            if basename is None:
                file_name = None
            else:
                file_name = basename + "imps_hm_" + z0 + z1 + t + str(i) + ".png"

            heat_map(fi, np.array(subset) + 1, self.feature_names, file_name)

    def save(self, filename, keep_dataset=False):
        if not keep_dataset:
            self.datset= []
        import cPickle
        cPickle.dump(self, open(filename, 'wb'), -1)



