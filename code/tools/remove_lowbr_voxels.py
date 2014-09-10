from neurosynth.base.dataset import Dataset
import neurosynth.base.imageutils as it

dataset = Dataset.load("../data/datasets/abs_topics_filt.pkl")

print "Filtering voxels..."

data = dataset.image_table.data.toarray()

voxel_mask = data.mean(axis=1) > 0.005

img = it.load_imgs('../masks/ward/30.nii.gz', dataset.masker)

good_voxels = img[voxel_mask]

it.save_img(good_voxels, "../masks/ward/30_masked.nii.gz", dataset.masker)