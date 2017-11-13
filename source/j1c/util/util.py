import numpy as np


def bounding_box(img):
    """
    Returns the z, y, x vectors that create a bounding box
    of a mask.
    """
    z = np.any(img, axis=(1, 2))
    y = np.any(img, axis=(0, 2))
    x = np.any(img, axis=(0, 1))

    zmin, zmax = np.where(z)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    xmin, xmax = np.where(x)[0][[0, -1]]

    return (zmin, zmax + 1), (ymin, ymax + 1), (xmin, xmax + 1)


def get_uniques(ar):
    """
    Returns an ordered numpy array of unique integers in an array.
    This runs about four times faster than numpy.unique().

    Parameters
    ----------
    ar : array_like
        Input array. This will be flattened.

    Returns
    -------
    uniques : ndarray
        The sorted unique values.
    """
    bins = np.zeros(np.max(ar) + 1, dtype=int)
    bins[ar.ravel()] = 1
    uniques = np.nonzero(bins)[0]

    return uniques
