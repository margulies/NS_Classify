#!/usr/bin/python
# -*- coding: utf-8 -*-

from neurosynth.analysis import classify
import numpy as np
from scipy import stats

import itertools
from .. import tools
import re
import os.path as path
from tempfile import mkdtemp

from sklearn.preprocessing import binarize


def classify_parallel(args):
    (classifier, scoring, filename, feat_select,
     comp_dims, class_weight), index = args

    X, y = np.memmap(filename, dtype='object', mode='r',
                     shape=comp_dims)[index]

    output = classify.classify(
        X, y, classifier=classifier, cross_val='4-Fold',
        class_weight=class_weight, scoring=scoring, feat_select=feat_select)

    output['index'] = index

    # Remember to add vector to output that keeps track of seleted features to
    # asses stability
    return output


class GenericClassifier:

    def __init__(self, dataset, mask_img, classifier=None,
                 thresh=0.08, cv='4-Fold', memsave=False, thresh_low=None, remove_overlap=True):
        """
        thresh_low - if OvA then this is the threshold for the rest of the brain


        """

        self.mask_img = mask_img
        self.classifier = classifier
        self.dataset = dataset
        self.thresh = thresh
        self.cv = cv
        self.memsave = memsave
        self.c_data = None
        self.thresh_low = thresh_low
        self.remove_overlap = remove_overlap

    def load_mask_data(self, features=None):
        """ Loads ids and data for each individual mask """
        from neurosynth.analysis.reduce import average_within_regions

        # ADD FEATURE TO FILTER BY FEATURES
        masks_by_studies = average_within_regions(
            self.dataset, self.mask_img, threshold=self.thresh)

        study_ids = self.dataset.feature_table.data.index

        print "Loading data from neurosynth..."

        pb = tools.ProgressBar(len(list(masks_by_studies)), start=True)

        self.ids_by_masks = []
        self.data_by_masks = []
        for mask in masks_by_studies:

            m_ids = study_ids[np.where(mask == True)[0]]
            self.ids_by_masks.append(m_ids)
            self.data_by_masks.append(self.dataset.get_feature_data(ids=m_ids, features=features))
            pb.next()

        self.mask_num = masks_by_studies.shape[0]

    def initalize_containers(self, features, feat_select, dummy):
        """ Makes all the containers that will hold feature importances, etc """

        # Move to an init_containers function
        self.class_score = tools.mask_diagonal(
            np.ma.masked_array(np.zeros(self.comp_dims)))

        if self.memsave is False:
            self.fit_clfs = np.empty(self.comp_dims, object)

        if features:
            self.feature_names = features
        else:
            # If features leater get selected this is not correct but must be
            # updated
            self.feature_names = self.dataset.get_feature_names()

        if feat_select is not None:
            if re.match('.*best', feat_select) is not None:
                self.n_features = int(feat_select.split('-')[0])
            self.features_selected = np.empty(
                self.comp_dims, object)
        else:
            self.n_features = len(self.feature_names)

        if dummy is not None:
            self.dummy_score = tools.mask_diagonal(
                np.ma.masked_array(np.zeros(self.comp_dims)))
        else:
            self.dummy_score = None

        self.predictions = np.empty(self.comp_dims, object)

        if dummy is not None:
            self.dummy_predictions = np.empty(self.comp_dims, object)

        # Make feature importance grid w/ masked diagonals

        self.feature_importances = tools.mask_diagonal(np.ma.masked_array(np.zeros(self.comp_dims + (self.n_features, ))))

    def classify(self, features=None, scoring='accuracy', X_threshold=None, feat_select=None, processes=1, class_weight='auto', dummy=None):
        if self.c_data is None:
            self.load_data(features, X_threshold)
            self.initalize_containers(features, feat_select, dummy)

        print "Classifying..."
        pb = tools.ProgressBar(len(list(self.comparisons)), start=True)

        if processes > 1:
            from multiprocessing import Pool
            pool = Pool(processes=processes)
        else:
            pool = itertools

        try:
            filename = self.c_data.filename

            for output in pool.imap(
                classify_parallel, itertools.izip(
                    itertools.repeat(
                        (self.classifier, scoring, filename, feat_select, self.comp_dims, class_weight)),
                    self.comparisons)):

                index = output['index']
                self.class_score[index] = output['score']
                if self.memsave is False:
                    self.fit_clfs[index] = output['clf']

                try:
                    self.feature_importances[index] = output['clf'].clf.coef_[0]
                except AttributeError:
                    try:
                        self.feature_importances[index] = output['clf'].clf.feature_importances_
                    except AttributeError:
                        pass

                if feat_select:
                    self.features_selected[index] = output['features_selected']

                self.predictions[index] = output['predictions']

                if dummy is not None:
                    from sklearn.dummy import DummyClassifier

                    X, y = self.c_data[index]
                    output = classify.classify(
                        X, y, classifier=DummyClassifier(strategy=dummy), cross_val='4-Fold',
                        class_weight=class_weight, scoring=scoring, feat_select=feat_select)

                    self.dummy_score[index] = output['score']
                    self.dummy_predictions[index] = output['predictions']

                pb.next()
        finally:
            if processes > 1:
                pool.close()
                pool.join()

        if dummy is None:
            self.final_score = self.class_score
        else:
            self.final_score = self.class_score - self.dummy_score

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

        if ranking:
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


class OnevsallClassifier(GenericClassifier):

    def load_data(self, features, X_threshold):
        """ Load data into c_data """
        # Load data for each mask
        self.load_mask_data(features)

        filename = path.join(mkdtemp(), 'c_data.dat')
        self.c_data = np.memmap(filename, dtype='object',
                                mode='w+', shape=(self.mask_num))

        all_ids = self.dataset.image_table.ids

        # If a low thresh is set, then get ids for studies at that threshold
        if self.thresh_low is not None:
            ids_by_masks_low = []
            from neurosynth.analysis.reduce import average_within_regions
            masks_by_studies_low = average_within_regions(
                self.dataset, self.mask_img, threshold=self.thresh_low)
            for mask in masks_by_studies_low:
                m_ids = np.array(all_ids)[np.where(mask == True)[0]]
                ids_by_masks_low.append(m_ids)       

        # Set up data into c_data
        for num, on_ids in enumerate(self.ids_by_masks):

            # If a low threshold is set, then use that to filter "off_ids", otherwise use "on_ids"
            if self.thresh_low is not None:
                off_ids = list(set(all_ids) - set(ids_by_masks_low[num]))
            else:
                off_ids = list(set(all_ids) - set(on_ids))

            on_data = self.data_by_masks[num].dropna()

            off_data = self.dataset.get_feature_data(ids=off_ids).dropna()

            y = np.array([0] * off_data.shape[0] + [1] * on_data.shape[0])

            X = np.vstack((np.array(off_data), np.array(on_data)))

            from neurosynth.analysis.classify import regularize
            X = regularize(X, method='scale')

            if X_threshold is not None:
                X = binarize(X, X_threshold)

            self.c_data[num] = (X, y)

        if self.memsave:
            self.data_by_masks = []
            self.ids_by_masks = []

        self.comparisons = range(0, self.mask_num)

        self.comp_dims = (self.mask_num, )


class PairwiseClassifier(GenericClassifier):

    def load_data(self, features, X_threshold):
        """ Load data into c_data """
        # Load data for each mask
        self.load_mask_data(features)

        # Set up pair-wise data
        self.comparisons = list(
            itertools.combinations(range(0, self.mask_num), 2))

        filename = path.join(mkdtemp(), 'c_data.dat')
        self.c_data = np.memmap(filename, dtype='object',
                                mode='w+', shape=(self.mask_num, self.mask_num))

        # Filter data and arrange into c_data
        for pair in self.comparisons:

            x1 = self.data_by_masks[pair[0]]
            x2 = self.data_by_masks[pair[1]]

            reg1_ids = self.ids_by_masks[pair[0]]
            reg2_ids = self.ids_by_masks[pair[1]]

            if self.remove_overlap is True:
                reg1_set = list(set(reg1_ids) - set(reg2_ids))
                reg2_set = list(set(reg2_ids) - set(reg1_ids))

                x1 = np.array(x1)[np.where(np.in1d(reg1_ids, reg1_set))[0]]
                x2 = np.array(x2)[np.where(np.in1d(reg2_ids, reg2_set))[0]]

                reg1_ids = reg1_set
                reg2_ids = reg2_set
                
            y = np.array([0] * len(reg1_ids) + [1] * len(reg2_ids))

            X = np.vstack((x1, x2))

            if X_threshold is not None:
                X = binarize(X, X_threshold)

            from neurosynth.analysis.classify import regularize
            X = regularize(X, method='scale')

            self.c_data[pair] = (X, y)

        if self.memsave:
            self.data_by_masks = []
            self.ids_by_masks = []

        self.comp_dims = (self.mask_num, self.mask_num)


def rescore(clf, scoring_function, dummy=None):
    """" Rescores clf given a scoring function
    will need to fix to work w/ pairwise later """
    result = []

    if dummy is None:
        predictions = clf.predictions
    else:
        predictions = clf.dummy_predictions

    for reg in predictions:
        tmp = []
        for cvl in reg:
            tmp.append(scoring_function(*cvl))
        import numpy as np
        result.append(np.array(tmp).mean())

    return result