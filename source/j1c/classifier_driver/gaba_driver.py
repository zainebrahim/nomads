import pickle

import numpy as np


def get_cubes(raw_data, centroids):
    """
    Parameters
    ----------
    raw_data : dict
        Each key represents channel and value represents raw data
    centroids : list
        In [z, y, x] format
    """
    exclude_list = ['dapi', 'mbp']
    channels = [x for x in raw_data.keys() if x.lower() not in exclude_list]
    channels = list(raw_data.keys())
    max_size = raw_data[channels[0]].shape
    centroids = np.array(centroids)

    cube_size = (7, 4, 4)  # Results in 15 x 9 x 9 cubes
    out = []

    for row in centroids:
        cubes = []
        z, y, x = row

        for chan in channels:
            data = raw_data[chan]
            z_idx = (z - cube_size[0], z + cube_size[0])
            y_idx = (y - cube_size[1], y + cube_size[1])
            x_idx = (x - cube_size[2], x + cube_size[2])

            # Dont deal with cubes on the edge of data
            if (z[0] >= 0) and (y[0] >= 0) and (x[0] >= 0) and (
                    z[1] <= max_size[0]) and (y[1] <= max_size[1]) and (
                        x[1] <= max_size[2]):
                cube = data[z_idx[0]:z_idx[1], y_idx[0]:y_idx[1], x_idx[0]:
                            x_idx[1]]
                cubes.append(cube)
        # Flatten array
        out.append(np.array(cubes, dtype=np.uint8).ravel())

    return np.array(out, dtype=np.unint8)


def gaba_classifier_pipeline(raw_data, centroids):
    """
    Parameters
    ----------
    raw_data : dict
        Each key represents channel and value represents raw data
    centroids : list
        In [z, y, x] format
    """
    X = get_cubes(raw_data, centroids)

    components = np.load('./components.npy')

    # Some data managing
    if data.shape[1] > components.shape[1]:
        data = data[:, :components.shape[1]]
    elif data.shape[1] < components.shape[1]:
        components = components[:, :data.shape[1]]

    X = X @ components.T

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(X)

    return predictions