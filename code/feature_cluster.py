from base.classifiers import OnevsallContinuous
from neurosynth.analysis import reduce as nsr
from neurosynth.analysis import cluster
from neurosynth.base.dataset import Dataset
import numpy as np
from neurosynth.base.mask import Masker
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge


dataset = Dataset.load('../data/datasets/abs_60topics_filt_jul.pkl')
full_data = dataset.get_image_data(dense=False)

# cls = cluster.Clusterer(full_data, global_mask=dataset.masker)

n_regions = [20, 25, 30, 35, 40, 45, 50]

### For various resolutions
for resolution in [3, 2]:

	print resolution

	print "Downsampling..."
	cls_a = cluster.Clusterer(full_data, global_mask=dataset.masker, grid_scale = resolution, min_studies_per_voxel = 75)

	# low_res_data = (cls_a.data, cls_a.grid)

	# try: 
	# 	clf = OnevsallContinuous.load("../data/datasets/downsampled/all_vox_" + str(resolution) + "mm_Ridge.pkl")
	# except:
	# 	clf = OnevsallContinuous(dataset, low_res_data[0], classifier = Ridge())
	# 	clf.classify(scoring=r2_score)
	# 	clf.save("../data/clfs/downsampled/all_vox_" + str(resolution) + "mm_Ridge.pkl")

	# print "Empty downsampler"
	# # Make a downsampled masker
	# empty = np.empty(dataset.image_table.data.shape[0])
	# ds_masker = nsr.apply_grid(empty, masker = dataset.masker, scale = resolution)

	# # Replace data with feature importances from model
	# cls.data = clf.feature_importances
	# cls.grid = ds_masker[1]
	# cls.masker = Masker(ds_masker[1])

	print "Clustering..."
	# Cluster with features
	# print "Minik"
	# for i in n_regions:
	# 	print i
	# 	cls.output_dir = '../results/cluster/cls_' + str(resolution)+ 'mm_minik/'
	# 	cls.cluster(n_clusters = i, algorithm='minik')

	# print "Ward"
	# for i in n_regions:
	# 	print i
	# 	cls.output_dir = '../results/cluster/cls_' + str(resolution) + 'mm_ward/'
	# 	cls.cluster(n_clusters = i, algorithm ='ward')

	# Cluster with coactivation

	print "Coactivation clustering"
	# print "Minik"
	# for i in n_regions:
	# 	print i
	# 	cls_a.output_dir = '../results/cluster/cls_' + str(resolution) + 'mm_minik_coact/'
	# 	cls_a.cluster(n_clusters = i, algorithm='minik')
	print "Ward"
	for i in n_regions:
		print i
		cls_a.output_dir = '../results/cluster/cls_' + str(resolution) + 'mm_ward_coact_min75v/'
		cls_a.cluster(n_clusters = i, algorithm ='ward')