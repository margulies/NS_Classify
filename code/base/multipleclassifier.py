#!/usr/bin/python
# -*- coding: utf-8 -*-

from neurosynth.analysis import classify
import os
import itertools
import numpy as np
from nipype.interfaces import fsl
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.feature_selection import RFE

from scipy import stats
from sklearn.preprocessing import binarize

from multiprocessing import Pool
from sklearn.linear_model import LassoCV

import tools

from plotting import heat_map

from statistics import shannons

from scipy import sparse

from tempfile import mkdtemp
import os.path as path

import re


def LassoCV_parallel(args):
    c_data, pair = args

    index, _ = tools.get_index_path(pair)

    X, y = c_data[index]

    try:
        L = LassoCV().fit(X, y)
        fs = np.where(L.coef_ != 0)[0]
        alpha = L.alpha_
    except:
        fs = None
        alpha = None

    return fs, alpha, index


def get_ns_for_pairs(a_b_c):
    """Convert `f([1,2])` to `f(1,2)` call."""
    dataset, pair, thresh = a_b_c

    index, names = tools.get_index_path(pair)

    X, y = classify.get_studies_by_regions(dataset, names, thresh)

    n = np.bincount(y)
    return (index, n)


def classify_parallel(args):
    (classifier, param_grid, scoring, filename, feat_select, length), pair = args

    index, names = tools.get_index_path(pair)
    
    X, y = np.memmap(filename, dtype='object', mode='r',
                       shape=(length, length))[index]
    X = X.toarray()

    output = classify.classify(
        X, y, classifier=classifier, output='summary_clf', cross_val='4-Fold',
        class_weight='auto', scoring=scoring, param_grid=param_grid, feat_select=feat_select)

    output['clf'].fit(X, y) # Is this necessary?

    output['index'] = index
    output['names'] = names

    # Remember to add vector to output that keeps track of seleted features to asses stability
    return output

class MaskClassifier:

    def __init__(self, dataset, masks, classifier=GradientBoostingClassifier(), 
        thresh=0.08, param_grid=None, cv=None):

        self.masklist = zip(masks, range(0, len(masks)))

        self.mask_names = [os.path.basename(os.path.splitext(os.path.splitext(mask)[0])[0])
                           for mask in masks]

        self.mask_num = len(self.masklist)

        # Where to store differences, results, n's and dummy results

        self.class_score = np.zeros((self.mask_num, self.mask_num))
        self.dummy_score = np.zeros((self.mask_num, self.mask_num))

        self.classifier = classifier
        self.dataset = dataset
        self.thresh = thresh
        self.param_grid = param_grid

        # Fitted classifier
        self.fit_clfs = np.empty((self.mask_num, self.mask_num), object)

        filename = path.join(mkdtemp(), 'c_data.dat')
        self.c_data = np.memmap(filename, dtype='object',
                                mode='w+', shape=(self.mask_num, self.mask_num))

        self.cv = cv

        self.status = 0

        if isinstance(self.classifier, RFE):
            # Fitted classifier
            self.feature_ranking = np.empty(
                (self.mask_num, self.mask_num), object)
        else:
            self.feature_ranking = None

    def load_data(self, features, mask_pairs, X_threshold):
        """ Load data into c_data """

        print "Loading data..."
        pb = tools.ProgressBar(len(list(mask_pairs)))

        pb.next()

        # Load data
        for pair in mask_pairs:
            index, names = tools.get_index_path(pair)

            X, y = classify.get_studies_by_regions(
                self.dataset, names, threshold=self.thresh, features=features, regularization='scale')

            if X_threshold is not None:
                X = binarize(X, X_threshold)

            X = sparse.csr_matrix(X)

            self.c_data[index] = (X, y)

            pb.next()

    def classify(self, features=None, scoring='accuracy', X_threshold=None, feat_select=None, processes=4):

        mask_pairs = list(itertools.permutations(self.masklist, 2))

        if features:
            self.feature_names = features
        else:
            self.feature_names = self.dataset.get_feature_names()

        if feat_select is not None and re.match('.*-best', feat_select) is not None:
            self.n_features = feat_select.split('-')[0]
        else:
            self.n_features = len(self.feature_names)


        # Make feature importance grid w/ masked diagonals
        self.feature_importances = tools.mask_diagonal(
            np.ma.masked_array(np.zeros((self.mask_num,
                                         self.mask_num, self.n_features))))

        self.features_selected = np.empty(
            (self.mask_num, self.mask_num), object)

        self.load_data(features, mask_pairs, X_threshold)

        print "Classifying..."
        pb = tools.ProgressBar(len(list(mask_pairs)), start=True)

        pool = Pool(processes=processes)

        try:
            filename = self.c_data.filename

            for output in pool.imap_unordered(
                classify_parallel, itertools.izip(
                    itertools.repeat((self.classifier, self.param_grid, scoring, filename, feat_select, len(self.masklist))), 
                    mask_pairs)):

                index = output['index']
                names = output['names']
                self.class_score[index] = output['score']
                self.fit_clfs[index] = output['clf']

                if self.param_grid:  # Just get the FIs if you used a grid
                    try:
                        self.feature_importances[index] = self.fit_clfs[
                            index].best_estimator_.coef_[0]
                    except AttributeError:
                        try:
                            self.feature_importances[index] = self.fit_clfs[
                                index].feature_importances_
                        except AttributeError:
                            pass
                else:
                    try:
                        self.feature_importances[
                            index] = self.fit_clfs[index].clf.coef_[0]
                    except AttributeError:
                        try:
                            self.feature_importances[index] = self.fit_clfs[
                                index].feature_importances_
                        except AttributeError:
                            pass

                self.features_selected[index] = output['features_selected']
                
                self.dummy_score[index] = classify.classify_regions(
                    self.dataset, names,
                    method='Dummy', threshold=self.thresh)['score']

                pb.next()
        finally:
            pool.close()
            pool.join()

        self.class_score = np.ma.masked_array(self.class_score, self.class_score == 0)
        self.dummy_score = np.ma.masked_array(self.dummy_score, self.dummy_score == 0)

        self.final_score = self.class_score - self.dummy_score

        self.status = 1

    def calculate_ns(self):
        mask_pairs = list(itertools.combinations(self.masklist, 2))
        self.ns = np.ma.masked_array(
            np.empty((self.mask_num, self.mask_num, 2)), True)

        pb = tools.ProgressBar(len(list(mask_pairs)))

        for index, n in itertools.imap(get_ns_for_pairs, itertools.izip(itertools.repeat(self.dataset), mask_pairs, itertools.repeat(self.thresh))):
            self.ns[index] = n
            pb.next()

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

        fsl.ImageMaths(in_file=folder + '/0.nii.gz', op_string=' -add ' +
                       folder + '/1', out_file=folder + '/ongoing.nii.gz').run()

        if self.mask_num > 1:
            for n in range(2, self.mask_num):
                fsl.ImageMaths(in_file=folder + '/ongoing.nii.gz', op_string=' -add '
                               + folder + '/' + str(n), out_file=folder + '/ongoing.nii.gz').run()

        fsl.ImageMaths(in_file=folder + '/ongoing.nii.gz',
                       op_string=' -sub 1' + ' -thr 0', out_file=out_file).run()

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

        if index is not None:  # If None, take it out and get all
            fi = fi[index]

        # If not a tuple (i.e. integer or None), get mean
        if not isinstance(index, tuple):
            if index is None:
                fi = fi.mean(axis=0).mean(axis=0)
            else:
                fi = fi.mean(axis=0)

        if absolute:
            fi = np.abs(fi)

        if demeaned:
            [fi_all, names] = zip(
                *self.get_importances(None, sort=False, relative=False, absolute=absolute, ranking=ranking))
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

        [imps, names] = zip(
            *self.get_importances(index, absolute=absolute, ranking=ranking))

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

    def get_best_features(self, n, ranking=True):
        """ Gets the n best features across all comparisons from a RFE classifier """

        return self.get_importances(None, absolute=True, ranking=ranking)[-n:]

    def save_region_importance_plots(self, basename, thresh=20):
        for i in range(1, self.mask_num):
            self.plot_importances(
                i - 1, file_name=basename + "_imps_" + str(i) + ".png", thresh=thresh)
            self.plot_importances(
                None, file_name=basename + "_imps_overall.png", thresh=thresh)

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

        for i in subset:
            region_data = fi[subset.index(i)]

            if method == 'var':
                results.append(np.apply_along_axis(np.var, axis, region_data))

            elif method == 'shannons':
                results.append(
                    np.apply_along_axis(shannons, axis, region_data))

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
            fi.mask = fi < thresh
            t = "zt" + str(thresh) + "_"

        heat_map(fi, np.array(subset) + 1, self.feature_names,
                 basename + "imps_hm_" + z0 + z1 + t + "overall.png")

        for i in subset:

            fi = overall_fi[subset.index(i)].T

            if zscore_regions:
                fi = np.ma.masked_invalid(stats.zscore(fi, axis=0))
            if zscore_features:
                fi = stats.zscore(fi, axis=1)

            if thresh is not None:
                fi.mask = fi < thresh

            if basename is None:
                file_name = None
            else:
                file_name = basename + "imps_hm_" + \
                    z0 + z1 + t + str(i) + ".png"

            heat_map(fi, np.array(subset) + 1, self.feature_names, file_name)

    def save(self, filename, keep_dataset=False, keep_cdata=False, keep_clfs=False):
        if not keep_dataset:
            self.dataset = []
        if not keep_cdata:
            self.c_data = []

        if not keep_clfs:
            self.fit_clfs = []
        import cPickle
        cPickle.dump(self, open(filename, 'wb'), -1)

    @classmethod
    def load(cls, filename):
        """ Load a pickled Dataset instance from file. """
        import cPickle
        return cPickle.load(open(filename, 'rb'))
