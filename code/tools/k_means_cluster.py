from neurosynth.base.dataset import Dataset
from sklearn.cluster import KMeans
from sklearn.cluster import Ward

import numpy as np
from neurosynth.base.imageutils import save_img
from scipy import sparse

dataset = Dataset.load("../data/datasets/abs_topics_filt.pkl")

print "Filtering voxels..."

data = dataset.image_table.data.toarray()

voxel_mask = data.mean(axis=1) > 0.0135

good_voxels = data[voxel_mask]

good_voxels = sparse.csr_matrix(good_voxels)

for i in [20, 30, 40, 50]:
	print "Clustering..."

	print i

	k_means = KMeans(init='k-means++', n_clusters=i, n_jobs=16)
	k_means.fit(good_voxels)

	# ward = Ward(n_clusters=30)
	# ward.fit(good_voxels)

	print "Stretching clustering results..."
	cluster_voxels = np.zeros(voxel_mask.shape)
	cluster_voxels[voxel_mask] = k_means.labels_ + 1

	print "Saving image..."
	save_img(cluster_voxels, "k_means_" + str(i) + "_0135.nii.gz", dataset.masker)