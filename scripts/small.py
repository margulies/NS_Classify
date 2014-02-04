
import numpy as np
from scipy import stats
def get_regions_frequency(clf):
	region_frequency = np.array([classify.get_studies_by_regions(clf.dataset, [mask[0]], threshold=clf.thresh, features=reduced_topics_2, regularization=None)[0].mean(axis=0) for mask in clf.masklist])

# stats.pearsonr(region_frequency.flatten(), clf.feature_importances.mean(axis=0).flatten())


def calc_fi_progression(clf):
	""" Generates a matrix of shape regions x regions x features x number of trees. Calculates feature importance for each feature over time. 
	Features useful early should be more general while those later shoudl differentiate more difficult cases"""

	fis =  np.empty(clf.feature_importances.shape + (clf.fit_clfs[0,1].n_estimators,))

	for i in range(0, clf.mask_num):
		for j in range(0, clf.mask_num):
			if i == j:
				fis[i, j] = None
			else:
				fis[i, j] = np.apply_along_axis(lambda x: x[0].feature_importances_, 1, clf.fit_clfs[i, j].estimators_).T


	clf.fi_x_estimators = np.ma.masked_array(fis, mask=np.isnan(fis))

# np.apply_along_axis(lambda x: stats.pearsonr(x, np.arange(0, clf.fit_clfs[1, 2].n_estimators))[0], 3, clf.fi_x_estimators)

def calc_fi_progression_2(clf):
	""" Generates a matrix of shape regions x regions x features x number of trees. Calculates feature importance for each feature over time. 
	Features useful early should be more general while those later shoudl differentiate more difficult cases"""
	import pandas

	def get_fis(fit_clf):
		if fit_clf is None:
			return None
		else:
			return np.apply_along_axis(lambda x: x[0].feature_importances_, 1, fit_clf.estimators_)

	get_fis = np.vectorize(get_fis)

	fis = get_fis(clf.fit_clfs)

	clf.fi_x_estimators = np.ma.masked_array(fis, mask=pandas.isnull(fis))

def calc_avg_pdp(clf):
	from sklearn.ensemble.partial_dependence import partial_dependence
	import pandas

	pdps =  np.empty(clf.fit_clfs.shape + (clf.feature_importances.shape[2], 2, 100))

	for i in range(0, clf.mask_num):
		for j in range(0, clf.mask_num):

			if i == j:
				pdps[i, j] = None
			else:
				for feature in range(0, clf.feature_importances.shape[2]):
					pdp, a = partial_dependence(clf.fit_clfs[i, j], [feature], X=clf.c_data[i, j][0]) 
					pdps[i, j, feature] = [pdp[0], a[0]]

	clf.pdps = np.ma.masked_array(pdps, mask= pandas.isnull(pdps))

