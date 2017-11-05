import matplotlib.pyplot as plt
import numpy as np


def graph_performance(absolute=False, *args, **kwds):
    """
    Graphs quantitative performance of an algorithm based on predictions
    per ground truth and ground truth per predictions.

    If keyword arguments are given, then the label names are taken from
    the keywords. If arguments are passed in with no keywords, then the
    labels are label_0, label_1, etc.

    Parameters
    ----------
    absolute : boolean, optional
        If True, the data will be presented in absolute frequencies instead of
        relative frequencies.
    args : Arguments
        Overlap dictionaries to plot. Since it is not possible to know what each
        overlap dictionary refers to, the data will be plotted with labels
        "label_0", "label_1", and so on.
    kwds : Keyword Arguments
        Overlap dictionaries to plot. Data will be plotted with the keyword labels.

    Examples
    --------
    >>> overlap_1 = compute_overlap_array(predictions_1, gt_1)
    >>> overlap_2 = compute_overlap_array(predictions_2, gt_2)
    >>> kwds = {'Label One': overlap_1, 'Label Two': overlap_2}
    >>> graph_performance(**kwds)
    """

    labeldict = kwds

    for i, val in enumerate(args):
        key = 'label_%d' % i
        if key in labeldict.keys():
            raise ValueError(
                "Cannot use un-named variables and keyword %s" % key)
        labeldict[key] = val

    labels = list(labeldict.keys())
    overlaps = [labeldict[label] for label in labels]

    #Global settings
    plt.rc(('xtick', 'ytick'), labelsize=14)
    plt.rc('axes', titlesize=16, labelsize=14)

    #Create fig and ax
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.9), sharey=True)

    #Titles for figure and two subplots
    ax[0].set_title('Number of Predictions Per Synapse')
    ax[1].set_title('Number of Synapses Per Prediction')

    #Set xaxis tick labels
    ax[0].set_xticklabels(['False Negatives', 'Correct', 'Double Counted'])
    ax[1].set_xticklabels(['False Positives', 'Correct', 'Merged'])

    #Set yaxis label and range accordingly
    if not absolute:
        ax[0].set_ylabel("% of population")
        ax[0].set_ylim(0, 100)
    else:
        ax[0].set_ylabel("Total number")

    #Plot stuff
    for i, label in enumerate(labels):
        #Graphing settings
        width = 0.8 / len(labels)
        offset = np.arange(0, width * len(labels), width)

        for j, key in enumerate(['predictionPerGt', 'gtPerPrediction']):
            counts = np.bincount(overlaps[i][key])

            #Data validations
            if len(counts) == 2:
                counts = np.append(counts, [0])
            elif len(counts) > 3:
                counts = np.append(counts[0:2], [np.sum(counts[2:])])

            if not absolute:
                data = [x / sum(counts) * 100 for x in counts]
            else:
                data = counts

            ax[j].bar(np.arange(3) + offset[i], data,
                      width=width, align='edge', label=label)
            ax[j].set(xticks=np.arange(3) + .4)

    #Finishing touches
    fig.tight_layout()
    plt.legend()

    return fig
