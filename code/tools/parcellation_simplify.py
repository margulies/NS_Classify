# This script takes a parcellation that may not have continous numbers, 
# removes parcels below some size and reorders in order 
# while also outputting the region and community number
from neurosynth.base.mask import Masker
from neurosynth.base import imageutils
import numpy as np
import csv

min_vox = 300

file = '../masks/Andy/aal_MNI_V4.nii'
outfile = '../masks/Andy/aal_MNI_V4_' + str(min_vox) + '.nii'


# Load image with masker
masker = Masker(file)
img = imageutils.load_imgs(file, masker)

# How many levels in the original image
print "Original shape:"
print np.bincount([int(vox) for vox in img]).shape

# Get how many voxels per level and calc those that pass min_vox
count = np.bincount(img.astype('int').squeeze())
non_0_ix = np.where(count >= min_vox)[0]
zero_ix = np.where(count < min_vox)[0]

# Remove those not in a good community
bad = list(set(zero_ix))

# Remove
for value in bad:
	np.place(img, img == value, [0])

non_0_ix.sort()

# Translate numbers to continous 
translated = zip(range(1, non_0_ix.shape[0]+1), list(non_0_ix))

for pair in translated:
	np.place(img, img == pair[1], pair[0])

print "New shape:"
print np.bincount([int(vox) for vox in img]).shape
imageutils.save_img(img, outfile, masker)


# Write key
with open('../masks/Andy/parcels_' + str(min_vox) + '_key.csv', 'w') as file:
	writer = csv.writer(file)
	writer.writerow(['new', 'original'])
	for pair in translated:
		writer.writerow(list(pair))




