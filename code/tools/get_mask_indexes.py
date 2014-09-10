def get_mask_ix_dict(clf):
	import re
	masks, indexes = zip(*clf.masklist)
	mask_nums = [int(re.findall("([0-9]+).nii.gz$", p)[0]) for p in masks]

	return dict(zip(mask_nums, indexes))