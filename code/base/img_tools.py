from neurosynth.base import imageutils 
from neurosynth.base import mask 
import numpy as np

def remove_value(infile, vals_rm, outfile):
	masker = mask.Masker(infile)
	img = imageutils.load_imgs(infile, masker)

	# Remove value
	for val in vals_rm:
		np.place(img, img == val, [0])

	# Save
	imageutils.save_img(img, outfile, masker)
