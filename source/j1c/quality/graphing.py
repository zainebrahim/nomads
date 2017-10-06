import matplotlib.pyplot as plt
import numpy as np


def graph_performance(overlap_dict, normed=True):
    """
    Graphs quantitative performance of an algorithm based on predictions
    per gt and gt per predictions. 

    Parameters
    ----------
    overlap_dict : dict
        Dictionary holding information predictions per gt and 
        gt per predictions calculated from source/bstadt/Quality/Quality.py
    normed : boolean, optional
        If False, the data will be presented in absolute frequencies instead of
        relative frequencies
    """
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
    if normed:
        ax[0].set_ylabel("% of population")
        ax[0].set_ylim(0, 100)
    else:
        ax[0].set_ylabel("Total number")

    #Plot stuff
    for i, value in enumerate(overlap_dict.values()):
        counts = np.bincount(value)

        #Check if there are values > 2. If yes, put in 2 bin.
        if len(counts) > 3:
            counts = counts[0:2] + [np.sum(counts[2:])]

        if normed:
            data = [x / sum(counts) * 100 for x in counts]
        else:
            data = counts

        ax[i].bar(range(3), data, width=0.8, align='center')
        ax[i].set(xticks=range(3))

    #Finishing touches
    fig.tight_layout()

    return fig
