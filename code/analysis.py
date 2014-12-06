#!/usr/bin/python
# -*- coding: utf-8 -*-

# import glob
import sys
import datetime
import csv
# import os

# from random import shuffle

# from sklearn.linear_model import BayesianRidge
# from sklearn.linear_model import ARDRegression

# from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Ridge
# from sklearn.linear_model import RidgeClassifierCV
# from sklearn.linear_model import ElasticNet
# from sklearn.linear_model import ElasticNet


# from sklearn.linear_model import Lasso

# from sklearn.linear_model import LassoCV


from base.classifiers import PairwiseClassifier
from base.classifiers import OnevsallClassifier
from base.classifiers import OnevsallContinuous
from base.classifiers import PairwiseContinuous


from base.tools import Logger
from base.pipelines import pipeline

from neurosynth.base.dataset import Dataset

import numpy as np

now = datetime.datetime.now()

# Setup

#Get old features
# with open('../data/unprocessed/original/features.txt') as f:
#     reader = csv.reader(f, delimiter="\t")
#     old_features = list(reader)[0][1:]

# dataset_topics = Dataset.load('../data/pickled_topics.pkl')
# dataset_topics_40 = Dataset.load('../data/dataset_topics_40.pkl')


# Mask list for various analysises
# yeo_7_masks =  glob.glob('../masks/Yeo_JNeurophysiol11_MNI152/standardized/7Networks_Liberal_*')
# yeo_17_masks =  glob.glob('../masks/Yeo_JNeurophysiol11_MNI152/standardized/17Networks_Liberal_*')

# Neurosynth cluser masks

# Mask lists for 10-100 craddock
# craddock_dir = "../masks/craddock/scorr_05_2level/"
# craddock_masks = [["craddock_" + folder, glob.glob(craddock_dir + folder + "/*")]
#                   for folder in os.walk(craddock_dir).next()[1]]


# Import reduced word features
# wr = csv.reader(open('../data/reduced_features.csv', 'rbU'), quoting=False)
# reduced_features = [word[0] for word in wr]
# reduced_features = [word[2:-1] for word in reduced_features]

# Import reduced topic features
# twr = csv.reader(open('../data/reduced_topic_keys.csv', 'rbU'))
# reduced_topics = ["topic_" + str(int(topic[0])) for topic in twr]

# reduced_topics_2 = ["topic_" + str(int(topic[0]))
# for topic in csv.reader(open('../data/topic_notjunk.txt', 'rbU'),
# quoting=False)]

# junk_topics_2 = list(set(dataset_topics_40.get_feature_names())
#                      - set(reduced_topics_2))

# features = dataset.get_feature_names()

# x = dataset_topics_40.feature_table.data.toarray()
# x[x < 0.05] = 0
# dataset_topics_40_thresh = dataset_topics_40
# dataset_topics_40_thresh.feature_table.data = sparse.csr_matrix(x)


# dataset_abstracts = Dataset.load("../data/datasets/dataset_abs_words_pandas.pkl")
# abs_features = dataset_abstracts.get_feature_names()
# filtered_abs_features = list(set(abs_features) & set(old_features))

# d_abs_topics = Dataset.load('../data/datasets')

d_abs_topics_filt = Dataset.load('../data/datasets/abs_60topics_filt_jul.pkl')
cognitive_topics = ['topic' + topic[0] for topic in csv.reader(
	open('../data/unprocessed/abstract_topics_filtered/topic_sets/topic_keys60-july_cognitive.txt', 'rU'), delimiter='\t') if topic[1] == "T"]

# Analyses
from sklearn.metrics import explained_variance_score


def complete_analysis(dataset, dataset_name, name, masklist, processes = 1, features=None):

    # for i in [10]:

		# pipeline(
		# 	OnevsallClassifier(dataset, masklist,
		# 		thresh=i, thresh_low = 0, memsave=False, classifier=RidgeClassifier()),
		# 	name + "_OvA_RidgeClassifier_DM_hard0_roc_" + dataset_name + "_tn_" + str(i), 
		# 	features=features, processes=processes, post = False, scoring = roc_auc_score, dummy='most_frequent')

		# pipeline(
	 #    	PairwiseClassifier(dataset, masklist,
	 #    		cv='4-Fold', thresh=i, memsave=True, remove_overlap = True, classifier=RidgeClassifier()),
	 #    	name + "_Pairwise_RidgeClassifier_roc_DM_" + dataset_name + "_tn_" + str(i), 
	 #    	features=features, processes=processes, post = False, scoring = roc_auc_score, dummy='most_frequent')

	pipeline(
		OnevsallContinuous(dataset, masklist, classifier=Ridge(), memsave=True),
		name + "_Ridge_" + dataset_name, 
		features=features, processes=processes, scoring = explained_variance_score)

	# pipeline(
	# 	PairwiseContinuous(dataset, masklist, classifier=Ridge(), memsave=True, remove_zero=True),
	# 	name + "_Pairwise_Ridge_rz_" + dataset_name, 
	# 	features=features, processes=processes, scoring = r2_score)

		# pipeline(
	 #    	PairwiseClassifier(dataset, masklist,
	 #    		cv='4-Fold', thresh=i, memsave=True, remove_overlap = False, classifier=RidgeClassifier()),
	 #    	name + "_Pairwise_RidgeClassifier_roc_overlap_DM_" + dataset_name + "_t_" + str(i), 
	 #    	features=features, processes=processes, post = False, scoring = roc_auc_score, dummy='most_frequent')


# Begin logging
sys.stdout = Logger("../logs/" + now.strftime("%Y-%m-%d_%H_%M_%S") + ".txt")

pr = 8;

try:


	# for topics in [60]:
	# 	d_abs_topics_filt = Dataset.load('../data/datasets/abs_' +str(topics) + 'topics_filt_jul.pkl')
	# 	for regions in [30]:
	# 		complete_analysis(d_abs_topics_filt, "abs_topics" + str(topics) + "_filt", "ward_" + str(regions), "../results/cluster_3mm_ward_coact/ClusterImages/Cluster_k" + str(regions) + ".nii.gz", processes = pr, features=None)
	# complete_analysis(d_abs_topics_filt, "abs_topics_filt", "all_voxels", None, processes = pr, features=None)

	# complete_analysis(d_abs_topics_filt, "abs_topics_filt", "aal", "../masks/Andy/aal_MNI_V4.nii", processes = pr, features=None)
	# complete_analysis(d_abs_topics_filt, "abs_topics_filt", "craddock_30", "../masks/craddock/scorr_05_2level/30/merged.nii.gz", processes = pr, features=None)
	# complete_analysis(d_abs_topics_filt, "abs_topics_filt", "craddock_40", "../masks/craddock/scorr_05_2level/40/merged.nii.gz", processes = pr, features=None)
	complete_analysis(d_abs_topics_filt, "abs_cog_topics", "wardmin75_30", "../results/cluster/cls_3mm_ward_coact_min75v/ward_k30/cluster_labels.nii.gz", processes = pr, features=cognitive_topics)

	# complete_analysis(d_abs_topics_filt, "abs_topics_filt", "ward_f20_test", "../results/cluster_3mm_ward/ClusterImages/Cluster_k20.nii.gz", processes = pr, features=None)


finally:
    sys.stdout.end()
