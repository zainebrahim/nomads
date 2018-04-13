import numpy as np
import pandas as pd

def get_aggregate_sum(synapse_centroids, data):
    z_max, y_max, x_max = data[next(iter(data))].shape
    data_dictionary = dict((key, []) for key in data.keys())
    for centroid in synapse_centroids:
        z, y, x = centroid
        z_lower = z - 11
        z_upper = z + 11
        y_lower = y - 11
        y_upper = y + 11
        x_lower = x - 11
        x_upper = x + 11
        # prob a better way but w/e tired rn
        # ignore boundary synapses
        if z_lower < 0 or z_upper >= z_max:
            continue
        if y_lower < 0 or y_upper >= y_max:
            continue
        if x_lower < 0 or x_upper >= x_max:
            continue
        for key in data.keys():
            data_dictionary[key].append(np.sum(data[key][z_lower:z_upper, y_lower:y_upper, x_lower: x_upper]))
    return data_dictionary
    
def get_data_frame(data_dict):
    df = pd.DataFrame(data_dict)
    df = df.loc[:, (df != 0).any(axis=0)]
    return df
    
