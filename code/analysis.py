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

from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV

# from sklearn.linear_model import Lasso

# from sklearn.linear_model import LassoCV


from base.classifiers import PairwiseClassifier
from base.classifiers import OnevsallClassifier

from base.tools import Logger
from base.pipelines import pipeline

from neurosynth.base.dataset import Dataset

# import numpy as np

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

dataset_abstracts_topics = Dataset.load("../data/datasets/dataset_abs_topics_pandas.pkl")



# Analyses

def complete_analysis(dataset, dataset_name, name, masklist, processes = 1, features=None):

    i = 0.1

    # pipeline(
    # 	PairwiseClassifier(dataset, masklist,
    # 		cv='4-Fold', thresh=i),
    # 	name + "_GB_DM_f1_" + dataset_name + "_t_" + str(i), 
    # 	features=features, processes=processes, post = False, scoring = 'f1', dummy = 'most_frequent')

    pipeline(
    	OnevsallClassifier(dataset, masklist,
    		cv='4-Fold', thresh=i, memsave = True, classifier=RidgeClassifier()),
    	name + "OvA_RidgeClassifier_DM_" + dataset_name + "_t_" + str(i), 
    	features=features, processes=processes, post = False, dummy = 'most_frequent')

    # pipeline(
    # 	PairwiseClassifier(dataset, masklist,
    # 		cv='4-Fold', thresh=i, classifier=LassoCV(max_iter=10000)),
    # 	name + "_LassoCV_DM_" + dataset_name + "_t_" + str(i), 
    # 	features=features, processes=processes, class_weight=None, post = False, scoring = 'accuracy', dummy = 'most_frequent')

# Begin logging
sys.stdout = Logger("../logs/" + now.strftime("%Y-%m-%d_%H_%M_%S") + ".txt")

try:
	complete_analysis(dataset_abstracts_topics, "abstract_topics", "ns_11", "../masks/ns_kmeans_all/kmeans_all_11.nii.gz", processes = 8, features=None)
	# complete_analysis(dataset_abstracts, "abstract_words", "ns_20", "../masks/ns_kmeans_all/kmeans_all_20.nii.gz", processes = 8, features=None)
	# complete_analysis(dataset_abstracts_topics, "abstract_topics", "ns_60", "../masks/ns_kmeans_all/kmeans_all_60.nii.gz", processes = 8, features=None)
	# complete_analysis(dataset_abstracts, "abstract_words", "ns_60", "../masks/ns_kmeans_all/kmeans_all_60.nii.gz", processes = 8, features=None)

finally:
    sys.stdout.end()
