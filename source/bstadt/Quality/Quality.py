import numpy as np
from scipy.stats import mode
from skimage.measure import label

def get_unique_overlap(foreground, background, i):
    '''
    Calculates the number of unique background labels in the foreground at i
    Does not count background label of 0
    '''
    unique = np.unique(np.multiply((foreground == i).astype(int), background))
    num_unique = len(unique)

    #0 is background label
    #should not count as a detection if
    #the prediction overlaps with the background
    if 0 in unique:
        num_unique-=1

    return num_unique


def compute_overlap_array(predictions, gt):

    predictionLabels = label(predictions)
    maxPredictionLabel = np.max(predictionLabels)

    gtLabels = label(gt)
    maxGtLabel = np.max(gtLabels)

    #first, look at how many unique predictions
    #overlap with a single gt synapse
    predictionPerGt = [get_unique_overlap(gtLabels, predictionLabels, i)\
                       for i in range(1, maxGtLabel)]


    #next, look at how many unique synapses overlap
    #with a single synapse prediction
    gtPerPrediction = [get_unique_overlap(predictionLabels, gtLabels, i)\
                       for i in range(1, maxPredictionLabel)]

    return {'predictionPerGt': predictionPerGt,
            'gtPerPrediction': gtPerPrediction}
