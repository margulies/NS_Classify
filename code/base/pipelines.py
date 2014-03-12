#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
from plotting import heat_map
from multipleclassifier import MaskClassifier


def post_processing(clf, basename):
	"""Given a MaskClassifier object and a basename, perform post processing"""

	print "_____________________"
	print "Descriptive results:"
	print "Overall accuracy: " + str(clf.final_score.mean())

	print "Mask averages: " + str(clf.get_mask_averages(precision=3))

	print "_____________________"
	print

    # Average classification accuracy brain map
	clf.make_mask_map(basename + "avg_class.nii.gz", clf.get_mask_averages())
	print "Made average accuracy map"

	print "Making consistency brain maps.."
	clf.make_mask_map(basename + "imps_shannons_0.nii.gz", clf.importance_stats(method='shannons'))
	clf.make_mask_map(basename + "imps_var_0.nii.gz", clf.importance_stats(method='var'))
	# clf.make_mask_map(basename + "imps_cor_0.nii.gz", clf.importance_stats(method='cor'))
	clf.make_mask_map(basename + "acc_shannons_0.nii.gz", clf.accuracy_stats(method='shannons'))
	clf.make_mask_map(basename + "clf_var_0.nii.gz", clf.accuracy_stats(method='var'))

	print "Making sparsity brain maps.."
	clf.make_mask_map(basename + "imps_shannons_1.nii.gz", clf.importance_stats(method='shannons', axis=1))
	clf.make_mask_map(basename + "imps_var_1.nii.gz", clf.importance_stats(method='var', axis=1))
	# clf.make_mask_map(basename + "imps_cor_1.nii.gz", clf.importance_stats(method='cor', axis=1))
	clf.make_mask_map(basename + "acc_shannons_1.nii.gz", clf.accuracy_stats(method='shannons'))
	clf.make_mask_map(basename + "clf_var_1.nii.gz", clf.accuracy_stats(method='var'))

	print "Making consistency heat maps..."
	heat_map(clf.importance_stats(method='shannons', axis=0, average=False).T,
		range(1, clf.mask_num), clf.feature_names, file_name=basename + "shannons_hm_0.png")
	heat_map(clf.importance_stats(method='var', axis=0, average=False).T,
		range(1, clf.mask_num), clf.feature_names, file_name=basename +
		"var_hm_0.png")

	print "Making sparsity heat maps..."
	heat_map(clf.importance_stats(method='shannons', axis=1, average=False).T,
		range(1, clf.mask_num), range(0, clf.mask_num), file_name=basename + "shannons_hm_1.png")
	heat_map(clf.importance_stats(method='var', axis=1, average=False).T,
		range(1, clf.mask_num), range(0, clf.mask_num), file_name=basename+"var_hm_1.png")
	# heat_map(clf.importance_stats(method='cor', axis=1, average=False).T,
	# 	range(0, clf.mask_num), clf.feature_names,
	# 	file_name=basename+"_cor_hm_1.png")

	print "Making region importance plots..."
	clf.save_region_importance_plots(basename)

	print "Making feature importance heatmaps..."
	clf.region_heatmap(basename)
	clf.region_heatmap(basename, zscore_regions=True)
	clf.region_heatmap(basename, zscore_features=True)
	clf.region_heatmap(basename, zscore_regions=True, zscore_features=True)

def pipeline(clf, name, features=None, retest=False, scoring='accuracy', X_threshold=None, processes=4, feat_select=None):

    print("Classifier: " + str(clf.classifier))

    # Classify and save $ print
    clf.classify(features=features, scoring=scoring,
                 X_threshold=X_threshold, feat_select=feat_select, processes=processes)

    # Make directory for saving
    basename = "../results/" + name
    add = ""
    while os.path.exists(basename + add):
        if add == "":
            add = "_1"
        else:
            add = "_" + str(int(add[1:]) + 1)
    basename = basename + add + "/"
    os.makedirs(basename)

    print basename

    # Save classifier
    clf.save(basename + "classifier.pkl")

    post_processing(clf, basename)

    # Test-restest reliability
    if retest:
    	print "Retesting..."
    	clf2 = clf
    	clf2.classify(features=features)
    	from scipy.stats import pearsonr
    	flat1 = clf.feature_importances.flatten()
    	flat2 = clf2.feature_importances.flatten()
    	cors = []
    	for num, item in enumerate(flat1):
    		if item is not None:
    			cors.append(pearsonr(item, flat2[num])[0])
    	avg_cor = np.array(cors).mean()
    	print "Pearson r of test and retest importances: " + str(avg_cor)
    return clf

def load_process(basename):
	""" Load and process classifier """
	try:
		clf = MaskClassifier.load(basename + "classifier.pkl")
	except IOError:
		raise Exception("No pickled classifer in this fodler")

	post_processing(clf, basename)