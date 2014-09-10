from neurosynth.base.mask import Masker
from neurosynth.base import imageutils
import numpy as np
import csv

min_vox = 100

file = '../masks/wash_u/Parcels.nii'
outfile = '../masks/wash_u/parcels_continuous_' + str(min_vox) + '_filt.nii'


with open('../masks/wash_u/ParcelCommunities.txt', 'rb') as comm_file:
    reader = csv.reader(comm_file, delimiter=' ')
    communities = list(reader)


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
bad = [int(row[0]) for row in communities if row[1] == '-1' or row[1] == '11.5']
non_0_ix = np.array(list(set(non_0_ix) - set(bad)))
bad = list(set.union(set(bad), set(zero_ix)))

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
with open('../masks/wash_u/parcels_continuous_' + str(min_vox) + '_key.csv', 'w') as file:
	writer = csv.writer(file)
	writer.writerow(['new', 'original'])
	for pair in translated:
		writer.writerow(list(pair))




