from tempfile import mkdtemp
import os.path as path

import numpy as np


filename = path.join(mkdtemp(), 'newfile.dat')

fp = np.memmap(filename, dtype='object', mode='w+', shape=(300, 300,300))

fp[2, 2, 2] = ("hi", )

del fp

newfp = np.memmap(filename, dtype='object', mode='r', shape=(300, 300,300))

print newfp[2, 2, 2]

print newfp



