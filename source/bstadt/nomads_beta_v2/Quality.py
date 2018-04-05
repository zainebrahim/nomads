import numpy as np
from skimage.measure import label
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu


def make_overlap_img(a, b):
    bin_a = a > threshold_otsu(a)
    bin_b = b > threshold_otsu(a)
    rgb_both = np.moveaxis(np.stack([bin_a, bin_b, np.zeros_like(bin_a)]), 0, -1)
    rgb_both = rgb_both.astype(float)
    return rgb_both

    
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


def compare(overlap_1, overlap_2, overlap_1_name, overlap_2_name, dataset_name, do_fig = True):
    arr = np.array(overlap_1['gtPerPrediction'])
    n_tp = np.sum([arr >= 1])
    n_fp = np.sum([arr == 0])
    arr = np.array(overlap_1['predictionPerGt'])
    n_fn = np.sum(np.ones_like(arr)) - n_tp

    arr = np.array(overlap_2['gtPerPrediction'])
    m_tp = np.sum([arr >= 1])
    m_fp = np.sum([arr == 0])
    arr = np.array(overlap_2['predictionPerGt'])
    m_fn = np.sum(np.ones_like(arr)) - m_tp

    if do_fig:
        plt.figure(figsize=(16, 10))
        plt.suptitle('Algorithm Performance on '+dataset_name)
        ax = plt.subplot(221)
        ax.set_xlabel('#of Predictions Per Synapse')
        ax.set_ylabel('# of Occurrences')

        arr = np.array(overlap_1['predictionPerGt'])

        freq = [np.sum([arr == 0]),
                np.sum([arr == 1]),
                np.sum([arr >= 2])]

        r1 = ax.bar([-.25, .75, 1.75], freq, .25)

        arr = np.array(overlap_2['predictionPerGt'])

        freq = [np.sum([arr == 0]),
                np.sum([arr == 1]),
                np.sum([arr >= 2])]

        r2 = ax.bar([0, 1, 2], freq, .25)



        ax.set_xticklabels(('','0', '', '1', '', '2+'))
        ax.legend((r1, r2), (overlap_1_name, overlap_2_name), loc='lower center', bbox_to_anchor=(1.7, -.9))



        ax = plt.subplot(222)
        ax.set_xlabel('# of Synapses Per Prediction')
        ax.set_ylabel('# of Occurrences')

        arr = np.array(overlap_1['gtPerPrediction'])

        freq = [np.sum([arr == 0]),
                np.sum([arr == 1]),
                np.sum([arr >= 2])]


        r1 = ax.bar([-.25, .75, 1.75], freq, .25)

        arr = np.array(overlap_2['gtPerPrediction'])

        freq = [np.sum([arr == 0]),
                np.sum([arr == 1]),
                np.sum([arr >= 2])]


        r2 = ax.bar([0, 1, 2], freq, .25)


        ax.set_xticklabels(('','0', '', '1', '', '2+'))

        ax = plt.subplot(223)
        ax.set_ylabel('# of Occurrences')

        ax.set_xticklabels(('','True\nPositive', '',  'False\nPositive', '', 'False\nNegative'))
        r1 = ax.bar([-.25, .75, 1.75], [n_tp, n_fp, n_fn], .25)
        r2 = ax.bar([0, 1, 2], [m_tp, m_fp, m_fn], .25)
        plt.show()
    return [[n_tp, n_fp, n_fn], [m_tp, m_fp, m_fn]]


def evaluate(overlap_1, dataset_name, do_fig = True):
    arr = np.array(overlap_1['gtPerPrediction'])
    n_tp = np.sum([arr >= 1])
    n_fp = np.sum([arr == 0])

    arr = np.array(overlap_1['predictionPerGt'])
    n_fn = np.sum(np.ones_like(arr)) - n_tp

    if do_fig:
        plt.figure(figsize=(16, 10))
        plt.suptitle('Algorithm Performance on '+dataset_name)
        ax = plt.subplot(221)
        ax.set_xlabel('#of Predictions Per Synapse')
        ax.set_ylabel('# of Occurrences')

        arr = np.array(overlap_1['predictionPerGt'])

        freq = [np.sum([arr == 0]),
                np.sum([arr == 1]),
                np.sum([arr >= 2])]

        r1 = ax.bar([0, 1, 2], freq, .25)

        ax.set_xticklabels(('','0', '', '1', '', '2+'))

        ax = plt.subplot(222)
        ax.set_xlabel('# of Synapses Per Prediction')
        ax.set_ylabel('# of Occurrences')

        arr = np.array(overlap_1['gtPerPrediction'])

        freq = [np.sum([arr == 0]),
                np.sum([arr == 1]),
                np.sum([arr >= 2])]

        r1 = ax.bar([0, 1, 2], freq, .25)

        ax.set_xticklabels(('','0', '', '1', '', '2+'))

        ax = plt.subplot(223)
        ax.set_ylabel('# of Occurrences')
        ax.set_xticklabels(('','True\nPositive', '',  'False\nPositive', '', 'False\nNegative'))
        r1 = ax.bar([0, 1, 2], [n_tp, n_fp, n_fn], .25)
        plt.show()


    return n_tp, n_fp, n_fn
