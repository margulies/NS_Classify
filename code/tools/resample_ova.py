#!/usr/bin/python
# -*- coding: utf-8 -*-

###
# This script shuffle the classification labels and reruns classification many times to get data to calculate a confidence interval around the null hypothesis

from sklearn.linear_model import RidgeClassifier
from base.classifiers import OnevsallClassifier
from neurosynth.base.dataset import Dataset
from sklearn.metrics import roc_auc_score
import pickle
from random import shuffle

def shuffle_data(classifier):
	for region in classifier.c_data:
		shuffle(region[1])


d_abs_topics_filt = Dataset.load('../data/datasets/abs_topics_filt_july.pkl')

results = []

clf = OnevsallClassifier(d_abs_topics_filt, '../masks/Ward/50.nii.gz', cv='4-Fold',
	 thresh=10, thresh_low=0, memsave=True, classifier=RidgeClassifier())
clf.load_data(None, None)
clf.initalize_containers(None, None, None)


for i in range(0, 500):
	shuffle_data(clf)
	clf.classify(scoring=roc_auc_score, processes=8, class_weight=None)
	results = list(clf.class_score) + results
	print(i),

pickle.dump(results, open('../results/ward_50_OvA_RidgeClassifier_DM_hard0_roc_abs_topics_filt_tn_10/resample_results.pkl', 'wb'))


