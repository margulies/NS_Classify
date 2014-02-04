execfile("regionalClassifier.py")
import glob
import sys
import datetime
import csv
from random import shuffle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import NuSVC
from sklearn.linear_model import RidgeClassifier

import cPickle

now = datetime.datetime.now()

#### Logging
class Logging():
    def __init__(self, logfile):
        self.stdout = sys.stdout
        self.log = open(logfile, 'w')
        self.old = sys.stdout
 
    def write(self, text):
        self.stdout.write(text)
        self.log.write(text)
        self.log.flush()
 
    def end(self):
        self.log.close()
        sys.stdout = self.old

    def flush(self):
    	self.stdout.flush()
    	self.log.flush()

    def show(self, text):
    	self.stdout.write(text)
    	self.stdout.flush()

#### Pipline
def pipeline(clf, name, features=None, retest=False, scoring='accuracy', X_threshold=None):

	print( "Classifier: " + str(clf.classifier))

	# Classify and save $ print 
	clf.classify(features=features, scoring=scoring, dummy=True, X_threshold=X_threshold)

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

	print "Mask averages: " + str(clf.get_mask_averages(precision=3))

	print "_____________________"
	print

	# Average classification accuracy brain map
	clf.make_mask_map(basename + "avg_class.nii.gz", clf.get_mask_averages())
	print "Made average accuracy map"

	print "Making consistency brain maps.."
	clf.make_mask_map(basename + "imps_shannons_0.nii.gz", clf.importance_stats(method='shannons'))
	clf.make_mask_map(basename + "imps_var_0.nii.gz", clf.importance_stats(method='var'))
	clf.make_mask_map(basename + "imps_cor_0.nii.gz", clf.importance_stats(method='cor'))
	clf.make_mask_map(basename + "acc_shannons_0.nii.gz", clf.accuracy_stats(method='shannons'))
	# clf.make_mask_map(basename + "clf_var_0.nii.gz", clf.accuracy_stats(method='var'))

	print "Making sparsity brain maps.."
	clf.make_mask_map(basename + "imps_shannons_1.nii.gz", clf.importance_stats(method='shannons', axis=1))
	clf.make_mask_map(basename + "imps_var_1.nii.gz", clf.importance_stats(method='var', axis=1))
	clf.make_mask_map(basename + "imps_cor_1.nii.gz", clf.importance_stats(method='cor', axis=1))
	clf.make_mask_map(basename + "acc_shannons_1.nii.gz", clf.accuracy_stats(method='shannons'))
	# clf.make_mask_map(basename + "clf_var_1.nii.gz", clf.accuracy_stats(method='var'))

	print "Making consistency heat maps..."
	heat_map(clf.importance_stats(method='shannons', axis=0, average=False).T, 
		range(1, clf.mask_num), clf.feature_names, file_name=basename + "shannons_hm_0.png")
	heat_map(clf.importance_stats(method='var', axis=0, average=False).T, 
		range(1, clf.mask_num), clf.feature_names, file_name=basename + "var_hm_0.png")

	# heat_map(clf.importance_stats(method='cor', axis=0, average=False).T, 
	# 	range(0, clf.mask_num), clf.feature_names, file_name=basename+"_cor_hm_0.png")

	print "Making sparsity heat maps..."
	heat_map(clf.importance_stats(method='shannons', axis=1, average=False).T, 
		range(1, clf.mask_num), range(0, clf.mask_num), file_name=basename + "shannons_hm_1.png")
	heat_map(clf.importance_stats(method='var', axis=1, average=False).T, 
		range(1, clf.mask_num), range(0, clf.mask_num), file_name=basename+"var_hm_1.png")
	# heat_map(clf.importance_stats(method='cor', axis=1, average=False).T, 
	# 	range(0, clf.mask_num), clf.feature_names, file_name=basename+"_cor_hm_1.png")

	# print "Making region importance plots..."
	# clf.save_region_importance_plots(basename)

	print "Making feature importance heatmaps..."
	clf.region_heatmap(basename)
	clf.region_heatmap(basename, zscore_regions=True)
	clf.region_heatmap(basename, zscore_features=True)
	clf.region_heatmap(basename, zscore_regions=True, zscore_features=True)


	# Test-restest reliability
	if retest:
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

		print "Pearson r of test and retest importances: " + str(avg_cor)

	return clf

### Setup

# Load regular database and topic database
dataset = Dataset.load('../data/dataset.pkl')
dataset_topics = Dataset.load('../data/pickled_topics.pkl')
dataset_topics_40 = Dataset.load('../data/pickled_topics_40.pkl')


#Mask list for various analysises
yeo_7_masks =  glob.glob('../masks/Yeo_JNeurophysiol11_MNI152/standardized/7Networks_Liberal_*')
yeo_17_masks =  glob.glob('../masks/Yeo_JNeurophysiol11_MNI152/standardized/17Networks_Liberal_*')

# Neurosynth cluser masks
ns_dir = "../masks/ns_kmeans_all/"
ns_kmeans_masks = [["ns_k_" + folder, glob.glob(ns_dir + folder +"/*")] for folder in os.walk(ns_dir).next()[1]]


# Mask lists for 10-100 craddock
craddock_dir = "../masks/craddock/scorr_05_2level/"
craddock_masks = [["craddock_" + folder, glob.glob(craddock_dir + folder +"/*")] for folder in os.walk(craddock_dir).next()[1]]


# Import reduced word features
wr = csv.reader(open('../data/reduced_features.csv', 'rbU'), quoting=False)
reduced_features = [word[0] for word in wr]
reduced_features = [word[2:-1] for word in reduced_features]

# Import reduced topic features
twr = csv.reader(open('../data/reduced_topic_keys.csv', 'rbU'))
reduced_topics  = ["topic_" + str(int(topic[0])) for topic in twr]

reduced_topics_2 = ["topic_" + str(int(topic[0])) for topic in csv.reader(open('../data/topic_notjunk.txt', 'rbU'), quoting=False)]

junk_topics_2 = list(set(dataset_topics_40.get_feature_names()) - set(reduced_topics_2))

features = dataset.get_feature_names()

from scipy import sparse
x = dataset_topics_40.feature_table.data.toarray()
x[x<0.05] = 0
dataset_topics_40_thresh = dataset_topics_40
dataset_topics_40_thresh.feature_table.data = sparse.csr_matrix(x)


dataset_abstracts = Dataset.load("../data/new_dataset/new_dataset.pkl")

#############
# Analyses  #
#############

# Begin logging
sys.stdout = Logging("../logs/" + now.strftime("%Y-%m-%d_%H_%M_%S") + ".txt")

def complete_analysis(name, masklist, features=None):

	i = 0.07 

	# for i in [0.01, 0.03, 0.07, 0.1]:
	# 	print "Thresh = " + str(i)

	# 	# print "Words"

	# 	# pipeline(MaskClassifier(dataset, masklist, thresh=i, 
	# 	# 	param_grid=None, cv='4-Fold'), name+"_GB_words_reduced_thresh_"+str(i), features=None)

	# 	# pipeline(MaskClassifier(dataset, masklist, thresh=i, classifier=LinearSVC(class_weight="auto"), 
	# 	# 	param_grid=None, cv='4-Fold'), name+"_SVM_words_reduced_thresh_"+str(i), features=None)

	# 	# print
	# 	print "Topics"

	pipeline(MaskClassifier(dataset_abstracts, masklist, param_grid=None, cv='4-Fold',thresh=i), 
		name+"_GB_abstract_words_t_"+str(i), features=features)

	# pipeline(MaskClassifier(dataset_topics_40, masklist, classifier=NuSVC(0.15), cv='4-Fold',thresh=i),
	# 	name+"_NuSVC_topics_reduced_t_"+str(i), features=features)

	# pipeline(MaskClassifier(dataset_topics_40, masklist, param_grid=None, classifier=RidgeClassifier(alpha = 0.1),
	# 	cv='4-Fold',thresh=i), name+"_Ridge_0.1_topics_t_"+str(i), features=features)
	# all_topics = list(dataset_topics_40.get_feature_names())
	# # shuffle(all_topics)
	# pipeline(MaskClassifier(dataset_topics_40, masklist, param_grid=None, classifier=RidgeClassifier(),
	# 	cv='4-Fold',thresh=i), name+"_Ridge_topics_f1_t_"+str(i), features=features, scoring = 'f1')

	pipeline(MaskClassifier(dataset_abstracts, masklist, param_grid=None, classifier=RidgeClassifier(),
		cv='4-Fold',thresh=i), name+"_Ridge_abstractwords_"+str(i), features=features)
	
	# pipeline(MaskClassifier(dataset_topics_40, masklist, param_grid=None, classifier=RidgeClassifier(),
	# 	cv='4-Fold',thresh=i), name+"_Ridge_Xt_0.1_topics_t_"+str(i), features=None, X_threshold = 0.1)

	# pipeline(MaskClassifier(dataset_topics_40, masklist, param_grid=None, classifier=RidgeClassifier(),
	# 	cv='4-Fold',thresh=i), name+"_Ridge_Xt_0.3_topics_t_"+str(i), features=None, X_threshold = 0.3)

	# pipeline(MaskClassifier(dataset_topics_40, masklist, param_grid=None,
	# 	cv='4-Fold',thresh=i), name+"_GB_Xt_0.2_topics_t_"+str(i), features=None, X_threshold = 0.2)

	# pipeline(MaskClassifier(dataset_topics_40, masklist, param_grid=None, classifier=RidgeClassifier(alpha = 10),
	# 	cv='4-Fold',thresh=i), name+"_Ridge_0.7_topics_t_"+str(i), features=features)

	# pipeline(MaskClassifier(dataset, masklist, param_grid=None, classifier=RidgeClassifier(alpha = 0.1),
	# 	cv='4-Fold',thresh=i), name+"_Ridge_terms_t_"+str(i), features=None)

	# pipeline(MaskClassifier(dataset, masklist, param_grid=None, classifier=RidgeClassifier(alpha = 1),
	# 	cv='4-Fold',thresh=i), name+"_Ridge_terms_red_t_"+str(i), features=reduced_features)

	# all_terms = dataset.get_feature_names()
	# shuffle(all_terms)
	# pipeline(MaskClassifier(dataset, masklist, param_grid=None, classifier=RidgeClassifier(alpha = 1),
	# 	cv='4-Fold',thresh=i), name+"_Ridge_terms_rand_t_"+str(i), features=all_terms[:50])



# complete_analysis("Yeo7", yeo_7_masks)
# # complete_analysis("Yeo17", yeo_17_masks)

# craddock_masks.pop(1)
# # for name, masks in craddock_masks:
# # 	complete_analysis(name, masks)

# features = dataset.get_feature_names()

# shuffle(features)

# reduced_topics_2.remove('topic_5')


# complete_analysis(*craddock_masks[141])



# End  Logging
sys.stdout.end()


# {'max_features': np.linspace(2, 40, 4).astype(int), 
# 		'n_estimators': np.round(np.linspace(5, 141, 4)).astype(int),'learning_rate': np.linspace(0.05, 1, 4).astype('float')}
#



	# for num, c in enumerate(clf.fit_clfs.flatten()):
	# 	flag = False
	# 	if c:
	# 		if not flag:
	# 			flag = True
	# 			res = c.best_params_
	# 			for item in res:
	# 				res[item] = [res[item]]
	# 		else:
	# 			for item in res:
	# 				res[item] = res[item].append(c.best_params_[item])

	# for item in res:
	# 	print item
	# 	print np.array(res[item]).mean()

