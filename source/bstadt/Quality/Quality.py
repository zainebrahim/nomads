import numpy as np
from skimage.measure import label

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


def get_unique_overlap(foreground, background, i):
    '''
    Calculates the number of unique background labels in the foreground at i
    Does not count background label of 0
    '''
    z, y, x = bounding_box(foreground == i)

    foreground = foreground[z[0]:z[1], y[0]:y[1], x[0]:x[1]]
    background = background[z[0]:z[1], y[0]:y[1], x[0]:x[1]]

    overlaps = np.multiply((foreground == i), background)
    uniques = get_uniques(overlaps)

    num_unique = len(uniques)

    #0 is background label
    #should not count as a detection if
    #the prediction overlaps with the background
    if 0 in uniques:
        num_unique -= 1

    return num_unique


def compute_overlap_array(predictions, gt, compare_annotations=False):
    """
    When comparing two annotation volumes, set compare_annotations to True.
    """
    if not compare_annotations:
        predictions = label(predictions)
        prediction_uniques = get_uniques(predictions)[1:]
    elif compare_annotations:
        prediction_uniques = get_uniques(predictions)[1:]

    gt_uniques = get_uniques(gt)[1:]

    #first, look at how many unique predictions
    #overlap with a single gt synapse
    predictionPerGt = [get_unique_overlap(gt, predictions, i)
                       for i in gt_uniques]

    #next, look at how many unique synapses overlap
    #with a single synapse prediction
    gtPerPrediction = [get_unique_overlap(predictions, gt, i)
                       for i in prediction_uniques]

    return {'predictionPerGt': predictionPerGt,
            'gtPerPrediction': gtPerPrediction}
