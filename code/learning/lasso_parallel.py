from sklearn.datasets import make_classification

from sklearn.linear_model import LassoCV

from sklearn.cross_validation import KFold

import multiprocessing

import itertools

def classify_parallel(args):
	(X, y), clf = args
	cver = KFold(len(y), 4)

	scores = []
	for train, test in cver:
	            X_train, X_test, y_train, y_test = X[
	                train], X[test], y[train], y[test]

	            # Train classifier
	            clf.fit(X_train, y_train)

	            # Test classifier
	            scores.append(clf.score(X_test, y_test))

	return scores

p = multiprocessing.Pool()

clf = LassoCV()

datasets = [make_classification() for i in range(0, 8)]

results = []
for output in p.imap(classify_parallel, itertools.izip(datasets, itertools.repeat(clf))):
	print output
	results.append(output)





