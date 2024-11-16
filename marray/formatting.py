import numpy as np


def as_masked_array(arr):
    # temporary: fix for CuPy
    # eventually: rewrite to avoid masked array
    data = np.asarray(arr.data)
    mask = np.asarray(arr.mask)
    return np.ma.masked_array(data, mask)
