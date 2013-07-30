execfile("regionalClassifier.py")
import glob

#### Pipline

def pipeline(clf, basename, features=None):

	clf.classify(features=features)
	print "Mask averages"
	clf.get_mask_averages()

	print "Making average map..."
	clf.make_average_map(basename)

	print "Making region importance plots..."
	clf.save_region_importance_plots(basename)

	print "Overall accuracy above baseline: " + str(clf.diffs.mean())

	print "Retesting..."

	clf2 = clf
	clf2.classify(features=reduced_features)

	from scipy.stats import pearsonr
	flat1 = clf.feature_importances.flatten()

	flat2 = clf2.feature_importances.flatten()

	cors = []
	for num, item in enumerate(flat1):
		if item is not None:
			cors.append(pearsonr(item, flat2[num])[0])

	avg_cor = np.array(cors).mean()


	print "Pearson r of test and retest SVM weights: " + str(avg_cor)

### Setup
dataset = Dataset.load('../data/dataset.pkl')

yeo_7_masklist =  glob.glob('../masks/Yeo_JNeurophysiol11_MNI152/standardized/7Networks_Liberal_*')


# Import reduced word features
import csv
readfile = open('../data/reduced_features.csv', 'rbU')
wr = csv.reader(readfile, quoting=False)
reduced_features = [word[0] for word in wr]
reduced_features = [word[2:-1] for word in reduced_features]

# Import reduced topic features
print "Yeo 7-Class"
yeo7 = MaskClassifier(dataset, masklist, param_grid=None, classifier=LinearSVC(class_weight="auto"), cv='4-Fold')
pipeline(yeo7, "../results/Yeo_SVMcoef", features=reduced_features)


# param_grid={'max_features': np.arange(2, 140, 44),
#         'n_estimators': np.arange(5, 141, 50),
#         'learning_rate': np.arange(0.05, 1, 0.1)}

