from neurosynth.base.dataset import Dataset

dataset = Dataset.load("../data/datasets/abs_topics_filt.pkl")
regions = '../masks/ns_kmeans_all/kmeans_all_11.nii.gz'
from base.tools import region_vox_baserates
