execfile("regionalClassifier.py")
import glob
import sys

#### Logging

class Logging():
    def __init__(self, logfile):
        self.stdout = sys.stdout
        self.log = open(logfile, 'w')
 
    def write(self, text):
        self.stdout.write(text)
        self.log.write(text)
        self.log.flush()
 
    def close(self):
        self.log.close()

    def flush(self):
    	self.stdout.flush()
    	self.log.flush()

    def show(self):
    	self.stdout.write(text)
    	self.stdout.flush()

#### Pipline
def pipeline(clf, basename, features=None, retest=False):

	print( "Classifier: " + str(clf.classifier))

	clf.classify(features=features)
	print "Mask averages"
	print(clf.get_mask_averages())

	print "Making average map..."
	clf.make_mask_map(basename + "AvgClass.nii.gz", clf.get_mask_averages())

	print "Average Shannon's Diversity"

	print(clf.get_mask_diversity())
	print "Making Shannon's Diversity Map..."

	clf.make_mask_map(basename + "Shannons.nii.gz", clf.get_mask_diversity())

	print "Making region importance plots..."
	clf.save_region_importance_plots(basename)

	print "Overall accuracy above baseline: " + str(clf.diffs.mean())

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

# Logging
import datetime
now = datetime.datetime.now()
oldstdout = sys.stdout
sys.stdout = Logging("../logs/" + now.strftime("%Y-%m-%d_%H_%M_%S") + ".txt")

# Load regular database and topic database
dataset = Dataset.load('../data/dataset.pkl')
dataset_topics = Dataset.load('../data/pickled_topics.pkl')

#Mask list for various analysises
yeo_7_masks =  glob.glob('../masks/Yeo_JNeurophysiol11_MNI152/standardized/7Networks_Liberal_*')
yeo_17_masks =  glob.glob('../masks/Yeo_JNeurophysiol11_MNI152/standardized/17Networks_Liberal_*')

# Mask lists for 10-100 craddock
craddock_dir = "../masks/craddock/scorr_05_2level/"
craddock_masks = [["craddock_" + folder, glob.glob(craddock_dir + folder +"/*")] for folder in os.walk(craddock_dir).next()[1]]


# Import reduced word features
import csv
readfile = open('../data/reduced_features.csv', 'rbU')
wr = csv.reader(readfile, quoting=False)
reduced_features = [word[0] for word in wr]
reduced_features = [word[2:-1] for word in reduced_features]

# Import reduced topic features
twr = csv.reader(open('../data/reduced_topic_keys.csv', 'rbU'))
reduced_topics  = ["topic_" + str(int(topic[0])+1) for topic in twr]

#############
# Analyses  #
#############

def complete_analysis(name, masklist):
	print name
	print "Words"

	pipeline(MaskClassifier(dataset, masklist, thresh=0.05, param_grid=None, cv='4-Fold'), 
		"../results/"+name+"_GB_words_", features=reduced_features)

	pipeline(MaskClassifier(dataset, masklist, thresh=0.05, classifier=LinearSVC(class_weight="auto"), 
		param_grid=None, cv='4-Fold'), "../results/"+name+"_SVM_words_", features=reduced_features)

	# print
	# print "Topics"
	# pipeline(MaskClassifier(dataset_topics, masklist, param_grid=None, cv='4-Fold'), 
	# 	"../results/"+name+"_GB_topics_", features=reduced_topics)

	# pipeline(MaskClassifier(dataset_topics, masklist, classifier=LinearSVC(class_weight="auto"), 
	# 	param_grid=None, cv='4-Fold'), "../results/"+name+"_SVM_topics_", features=reduced_topics)


# complete_analysis("Yeo7", yeo_7_masks)
# complete_analysis("Yeo17", yeo_17_masks)

craddock_masks.pop(1)
for name, masks in craddock_masks:
	complete_analysis(name, masks)


# End  Logging
sys.stdout.close()
sys.stdout = oldstdout



