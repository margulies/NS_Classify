#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from plotting import heat_map
from classifiers.pairwise import PairwiseClassifier
import classifiers.pairwise as mc


def post_processing(clf, basename):
	"""Given a PairwiseClassifier object and a basename, perform post processing"""


    # Average classification accuracy brain map
	mc.make_mask_map(clf, basename + "avg_class.nii.gz", mc.get_mask_averages(clf, ))
	print "Made average accuracy map"

	print "Making consistency brain maps.."
	mc.make_mask_map(clf, basename + "imps_shannons_0.nii.gz", mc.importance_stats(clf, method='shannons'))
	mc.make_mask_map(clf, basename + "imps_var_0.nii.gz", mc.importance_stats(clf, method='var'))
	# mc.make_mask_map(basename + "imps_cor_0.nii.gz", mc.importance_stats(method='cor'))
	mc.make_mask_map(clf, basename + "acc_shannons_0.nii.gz", mc.accuracy_stats(clf, method='shannons'))
	mc.make_mask_map(clf, basename + "clf_var_0.nii.gz", mc.accuracy_stats(clf, method='var'))

	print "Making sparsity brain maps.."
	mc.make_mask_map(clf, basename + "imps_shannons_1.nii.gz", mc.importance_stats(clf, method='shannons', axis=1))
	mc.make_mask_map(clf, basename + "imps_var_1.nii.gz", mc.importance_stats(clf, method='var', axis=1))
	# mc.make_mask_map(clf, basename + "imps_cor_1.nii.gz", mc.importance_stats(clf, method='cor', axis=1))
	mc.make_mask_map(clf, basename + "acc_shannons_1.nii.gz", mc.accuracy_stats(clf, method='shannons'))
	mc.make_mask_map(clf, basename + "clf_var_1.nii.gz", mc.accuracy_stats(clf, method='var'))

	# print "Making consistency heat maps..."
	heat_map(mc.importance_stats(clf, method='shannons', axis=0, average=False).T,
		range(1, clf.mask_num), clf.feature_names, file_name=basename + "shannons_hm_0.png")
	heat_map(mc.importance_stats(clf, method='var', axis=0, average=False).T,
		range(1, clf.mask_num), clf.feature_names, file_name=basename +
		"var_hm_0.png")

	print "Making sparsity heat maps..."
	heat_map(mc.importance_stats(clf, method='shannons', axis=1, average=False).T,
		range(1, clf.mask_num), range(0, clf.mask_num), file_name=basename + "shannons_hm_1.png")
	heat_map(mc.importance_stats(clf, method='var', axis=1, average=False).T,
		range(1, clf.mask_num), range(0, clf.mask_num), file_name=basename+"var_hm_1.png")

	print "Making feature importance heatmaps..."
	mc.region_heatmap(clf, basename)
	mc.region_heatmap(clf, basename, zscore_regions=True)
	mc.region_heatmap(clf, basename, zscore_features=True)
	mc.region_heatmap(clf, basename, zscore_regions=True, zscore_features=True)

def pipeline(clf, name, features=None, scoring='accuracy', X_threshold=None, processes=4, feat_select=None, class_weight = None, post = True, dummy = None):

    print("Classifier: " + str(clf.classifier))

    # Classify and save $ print
    clf.classify(features=features, scoring=scoring,
                 X_threshold=X_threshold, feat_select=feat_select, processes=processes, class_weight = class_weight, dummy = dummy)

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

    print "_____________________"
    print "Descriptive results:"
    print "Overall accuracy: " + str(clf.final_score.mean())

    if clf.dummy_score is not None:
    	print "Classifier averages: " + str(clf.class_score.mean())
    	print "Dummy averages: " + str(clf.dummy_score.mean())
    	from scipy import stats
    	print "Correlation between dummy and clf: " + str(stats.pearsonr(clf.class_score.flatten(), clf.dummy_score.flatten()))

    print "Mask averages: " + str(mc.get_mask_averages(clf, precision=3))

    print "_____________________"
    print

    with open(basename + "results.txt", 'w') as f:
    	f.write("Overall accuracy: " + str(clf.final_score.mean()))
    	f.write("Mask averages: " + str(mc.get_mask_averages(clf, precision=3)))

    	if clf.dummy_score is not None:
			f.write("Classifier averages: " + str(clf.class_score.mean()))
			f.write("Dummy averages: " + str(clf.dummy_score.mean()))
			from scipy import stats
			f.write("Correlation between dummy and clf: " + str(stats.pearsonr(clf.class_score.flatten(), clf.dummy_score.flatten())))

    if post:
	    post_processing(clf, basename)

    return clf

def load_process(basename):
	""" Load and process classifier """
	basename = basename + "/"
	try:
		clf = PairwiseClassifier.load(basename + "classifier.pkl")
	except IOError:
		raise Exception("No pickled classifer in this fodler")

	post_processing(clf, basename)