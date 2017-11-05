import numpy as np
from skimage.measure import label


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

    overlaps = np.multiply((foreground == i), background)
    uniques = get_uniques(overlaps)

    num_unique = len(uniques)

    #0 is background label
    #should not count as a detection if
    #the prediction overlaps with the background
    if 0 in uniques:
        num_unique -= 1

    return num_unique


def compute_overlap_array(predictions, gt):
    predictionLabels = label(predictions)
    maxPredictionLabel = np.max(predictionLabels)

    gt_uniques = get_uniques(gt)[1:]

    #first, look at how many unique predictions
    #overlap with a single gt synapse
    predictionPerGt = [get_unique_overlap(gt, predictionLabels, i)
                       for i in gt_uniques]

    #next, look at how many unique synapses overlap
    #with a single synapse prediction
    gtPerPrediction = [get_unique_overlap(predictionLabels, gt, i)
                       for i in range(1, maxPredictionLabel + 1)]

    return {'predictionPerGt': predictionPerGt,
            'gtPerPrediction': gtPerPrediction}
