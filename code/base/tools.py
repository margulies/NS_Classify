#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import numpy as np

class Logger():

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


class ProgressBar():

    def __init__(self, total, start=False):
        self.total = total
        self.current = 0.0
        self.last_int = 0
        if start:
            self.next()

    def update_progress(self, progress):
        display = '\r[{0}] {1}%'.format('#' * (progress / 10), progress)
        if sys.stdout.__class__.__name__ == 'Logging':
            sys.stdout.show(display)
        else:
            sys.stdout.write(display)
            sys.stdout.flush()

    def next(self):
        if not self.last_int == int((self.current) / self.total * 100):
            self.update_progress(int((self.current) / self.total * 100))
            self.last_int = int((self.current) / self.total * 100)

        if self.current == self.total:
            self.reset()
        else:
            self.current = self.current + 1

    def reset(self):
        print ""
        self.current = 0.0


def ix_subset(dataset, subset):
    return [i for i, x in enumerate(dataset.get_feature_names()) if x in subset]



def mask_diagonal(masked_array):
    """ Given a masked array, it returns the same array with the diagonals masked"""
    import numpy as np

    if len(masked_array.shape) == 3:
        i, j, k = np.meshgrid(
            *map(np.arange, masked_array.shape), indexing='ij')
        masked_array.mask = (i == j)
    elif len(masked_array.shape) == 2:
        i, j = np.meshgrid(
            *map(np.arange, masked_array.shape), indexing='ij')
        masked_array.mask = (i == j)

    return masked_array


def invert_y(y):
    import numpy as np
    y = y + 1
    y[y == 2] = 0

    return np.flipud(y)

def calculate_feature_corr(clf):
    import numpy as np
    from scipy import stats
    f_corr = np.empty(clf.feature_importances.shape)

    for i in range(0, clf.c_data.shape[0]):
        for j in range(0, clf.c_data.shape[1]):

            if i == j:
                f_corr[i, j] = None
            else:
                data, classes = clf.c_data[i, j]

                f_corr[i, j] = np.apply_along_axis(
                    lambda x: stats.pearsonr(x, classes), 0, data)[0]

    clf.feature_corr = np.ma.masked_array(f_corr, mask=np.isnan(f_corr))


def get_mask_ix_dict(clf):
    import re
    masks, indexes = zip(*clf.masklist)
    mask_nums = [int(re.findall("([0-9]+).nii.gz$", p)[0]) for p in masks]

    return dict(zip(mask_nums, indexes))

def region_vox_baserates(dataset, regions, threshold=False, remove_zero=True):
        """ Returns the baserate of every voxels within a set of regions

        Takes a Dataset and a Nifti image that defines distinct regions, and
        returns a list of matrices of voxels for each regions, where the value at each voxel is
        the mean activation. Each distinct ROI must have a
        unique value in the image; non-contiguous voxels with the same value will
        be assigned to the same ROI.

        Args:
            dataset: Either a Dataset instance from which image data are extracted, or a 
                Numpy array containing image data to use. If the latter, the array contains 
                voxels in rows and features/studies in columns. The number of voxels must 
                be equal to the length of the vectorized image mask in the 
            regions: An image defining the boundaries of the regions to use. Can be:
                1) A string name of the NIFTI or Analyze-format image
                2) A NiBabel SpatialImage
                3) A 1D numpy array of the same length as the mask vector in the Dataset's
                     current Masker.
            remove_zero: An optional boolean; when True, assume that voxels with value
            of 0 should not be considered as a separate ROI, and will be ignored.

        Returns:
            A list of 1D numpy array with voxels' base rates
        """

        import pandas as pd
        from scipy.sparse import vstack
        import itertools

        regions = dataset.masker.mask(regions)
        
        if threshold:
            dataset = dataset.get_image_data(dense=True)
        else:
            dataset = dataset.get_image_data(dense=False)

        start = 0
        if remove_zero:
                start = 1

        results = []
        labels = []
        for i in range(start, int(regions.max()) + 1):
                data = dataset[np.where(regions == i)[0]]

                if threshold:
                    data = data > threshold
                base_rates = data.mean(axis=1)
                results.append(base_rates)

                labels.append(base_rates.shape[0] * [i])

        results = vstack(results)
        labels = list(itertools.chain(*labels))

        results = pd.DataFrame([pd.Series(labels, dtype=object), pd.Series(results.toarray().flat)]).T
        results.columns = ['region', 'base_rate']

        return results


def region_n_vox(dataset, regions, remove_zero=True):
    mask = dataset.masker.mask(regions)

    results = np.bincount([int(num) for num in mask])

    if remove_zero:
        results = results[1:]

    return results

def copy_along_diagonal(array, data='upper_right', inverse = False):
    grid = np.meshgrid(*map(np.arange, array.shape), indexing='ij')

    g1, g2 = grid[0], grid[1]

    new_array = array.copy()

    if inverse:
        array = array * -1

    if data == 'upper_right':
        new_array[g1 > g2] = np.rot90(np.fliplr(array))[g2 < g1]
    else:
        new_array[g1 < g2] = np.rot90(np.fliplr(array))[g2 > g1]

    return new_array

def make_mask_map_ipython(data, infile):
    import tempfile
    from nbpapaya import Brain

    tmp_file = tempfile.mktemp(suffix=".nii")
    make_mask_map(data, infile, tmp_file)

    return Brain(tmp_file)

def make_mask_map(data, infile, outfile, index=None):
    from neurosynth.base.mask import Masker
    from neurosynth.base import imageutils

    # Load image with masker
    masker = Masker(infile)
    img = imageutils.load_imgs(infile, masker)

    data = list(data)

    if index is None:
        index = np.arange(0, len(data))
        rev_index = None
    else:
        all_reg = np.arange(0, img.max())
        rev_index = all_reg[np.invert(np.in1d(all_reg, index))]

    min_val = img.min()

    for num, value in enumerate(data):
        n = index[num]
        np.place(img, img == n + min_val, [value])

    if rev_index is not None:
        for value in rev_index:
            np.place(img, img == value + min_val, 0)

    img = img.astype('float32')

    imageutils.save_img(img, outfile, masker)

def make_mask_map_4d(data, infile, outfile):
    """ Make mask map with 4d dimeions
    data: values for levels in infile. Shape = [4th dimension, regions]
    infile: input file to replace levels with values
    outfile: output file name
    """
    from neurosynth.base.mask import Masker
    from neurosynth.base import imageutils
    from nibabel import nifti1

    data = np.array(data)

    # Load image with masker
    masker = Masker(infile)
    img = imageutils.load_imgs(infile, masker)

    header = masker.get_header()

    shape = header.get_data_shape()[0:3] + (data.shape[0],)
    header.set_data_shape(shape)

    result = []

    for t_dim, t_val in enumerate(data):
        result.append(img.copy())
        for num, value in enumerate(t_val):
            np.place(result[t_dim], img == num + 1, [value])

    result = np.hstack(result)

    header.set_data_dtype(result.dtype)  # Avoids loss of precision
    img = nifti1.Nifti1Image(masker.unmask(result).squeeze(), None, header)
    img.to_filename(outfile)
