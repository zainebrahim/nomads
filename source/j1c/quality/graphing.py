import matplotlib.pyplot as plt
import numpy as np


def graph_performance(overlaps, absolute=False, labels=None):
    """
    Graphs quantitative performance of an algorithm based on predictions
    per gt and gt per predictions.

    Parameters
    ----------
    overlaps : list or arr_like
        Dictionary holding information predictions per gt and
        gt per predictions calculated from source/bstadt/Quality/Quality.py
    absolute : boolean, optional
        If True, the data will be presented in absolute frequencies instead of
        relative frequencies.
    labels : list of str
        Required if you plot more than one algorithm performance.
    """
    if labels is None:
        labels = [None]

    assert isinstance(labels, list)
    assert len(overlaps) == len(
        labels), 'Labels required when graphing 2 or more performances.'

    #Global settings
    plt.rc(('xtick', 'ytick'), labelsize=14)
    plt.rc('axes', titlesize=16, labelsize=14)

    #Create fig and ax
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.9), sharey=True, legend=True)

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
    if None not in labels:
        plt.legend()

    return fig
