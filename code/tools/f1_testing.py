# f1 diagnostic

from sklearn.datasets import make_classification
from sklearn.linear_model import RidgeClassifier
from sklearn.cross_validation import cross_val_score
import pandas as pd
import numpy as np
from base.scoring import skew_f1

results = []
for i in np.arange(0.2, 1, .1):
	tries = []

	for t in range(0, 50):
		X, y = make_classification(n_samples = 1000, weights = [i, 1-i])

		clf = RidgeClassifier()
		tries.append(cross_val_score(clf, X, y, scoring = 'roc_auc').mean())


	results.append(np.array(tries).mean())

auc_results = pd.Series(results)