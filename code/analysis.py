#!/usr/bin/python
# -*- coding: utf-8 -*-

import glob
import sys
import datetime
import csv
import os

from random import shuffle
from sklearn.linear_model import RidgeClassifier

from base.multipleclassifier import MaskClassifier
from base.tools import Logger
from base.pipelines import pipeline

from scipy import sparse
from neurosynth.base.dataset import Dataset

now = datetime.datetime.now()

# Setup

# Load regular database and topic database
# dataset = Dataset.load('../data/dataset.pkl')
# dataset_topics = Dataset.load('../data/pickled_topics.pkl')
# dataset_topics_40 = Dataset.load('../data/dataset_topics_40.pkl')


# Mask list for various analysises
# yeo_7_masks =  glob.glob('../masks/Yeo_JNeurophysiol11_MNI152/standardized/7Networks_Liberal_*')
# yeo_17_masks =  glob.glob('../masks/Yeo_JNeurophysiol11_MNI152/standardized/17Networks_Liberal_*')

# Neurosynth cluser masks
# ns_dir = "../masks/ns_kmeans_all/"
# ns_kmeans_masks = [["ns_k_" + folder, glob.glob(ns_dir + folder +"/*")] for folder in os.walk(ns_dir).next()[1]]


# Mask lists for 10-100 craddock
craddock_dir = "../masks/craddock/scorr_05_2level/"
craddock_masks = [["craddock_" + folder, glob.glob(craddock_dir + folder + "/*")]
                  for folder in os.walk(craddock_dir).next()[1]]


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


dataset_abstracts = Dataset.load("../data/dataset_abstracts.pkl")


# Analyses

def complete_analysis(name, masklist, features=None):

    i = 0.07


    pipeline(
    	MaskClassifier(dataset_abstracts, masklist,
    		classifier=RidgeClassifier(), cv='4-Fold', thresh=i),
    	name + "_50b_Ridge_abstract_words_t_" + str(i), 
    	features=features, processes=1, feat_select="25-best")

    pipeline(
    	MaskClassifier(dataset_abstracts, masklist,
    		classifier=RidgeClassifier(), cv='4-Fold', thresh=i),
    	name + "_50b_Ridge_abstract_words_t_" + str(i), 
    	features=features, processes=16, feat_select="50-best")

    pipeline(
    	MaskClassifier(dataset_abstracts, masklist,
    		classifier=RidgeClassifier(), cv='4-Fold', thresh=i),
    	name + "_100b_Ridge_abstract_words_t_" + str(i), 
    	features=features, processes=16, feat_select="100-best")

    pipeline(
    	MaskClassifier(dataset_abstracts, masklist,
    		classifier=RidgeClassifier(), cv='4-Fold', thresh=i),
    	name + "_50b_Ridge_abstract_words_t_" + str(i), 
    	features=features, processes=16, feat_select="25-best")

    pipeline(
    	MaskClassifier(dataset_abstracts, masklist,
    		cv='4-Fold', thresh=i),
    	name + "_50b_GB_abstract_words_t_" + str(i), 
    	features=features, processes=16, feat_select="50-best")

    pipeline(
    	MaskClassifier(dataset_abstracts, masklist,
    		cv='4-Fold', thresh=i),
    	name + "_100b_GB_abstract_words_t_" + str(i), 
    	features=features, processes=16, feat_select="100-best")




#     all_terms = dataset.get_feature_names()
#     shuffle(all_terms)
#     pipeline(MaskClassifier(dataset, masklist, param_grid=None, classifier=RidgeClassifier(alpha = 1),
#     cv='4-Fold',thresh=i), name+"_Ridge_terms_rand_t_"+str(i),
#     features=all_terms[:50])


# Begin logging
sys.stdout = Logger("../logs/" + now.strftime("%Y-%m-%d_%H_%M_%S") + ".txt")
try:
    complete_analysis(*craddock_masks[0])
finally:
    sys.stdout.end()
