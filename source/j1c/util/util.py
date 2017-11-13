import numpy as np
from scipy.sparse import csr_matrix


def bounding_box2(img, fname=None):
    """
    Returns all z, y, x vectors that create a bounding box
    of a mask.

    Parameters
    ----------
    img : 3d-array like
        Sparsely labeled image (e.g. annotation)
    fname : string, optional
        File or file path to which the data is saved. The file
        is saved as a numpy array.

    Returns
    -------
    centroids : ndarray
        List of all centroids in format 
            (label, zmin, zmax, ymin, ymax, xmin, xmax)
    """

    sp_arr = csr_matrix(img.reshape((1, -1)))
    uniques = np.unique(sp_arr.data)

    bounds = np.zeros((len(uniques), 7), dtype=np.int)

    for i, label in enumerate(uniques):
        z, y, x = np.unravel_index(
            sp_arr.indices[sp_arr.data == label], img.shape)

        zmin, zmax = np.min(z), np.max(z)
        ymin, ymax = np.min(y), np.max(y)
        xmin, xmax = np.min(x), np.max(x)

        bounds[i] = label, zmin, zmax + 1, ymin, ymax + 1, xmin, xmax + 1

    if fname:
        np.save(fname, bounds)
    else:
        return bounds

'''
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
'''

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
