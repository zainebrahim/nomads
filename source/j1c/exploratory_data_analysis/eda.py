import itertools
import numpy as np
from scipy.sparse import csr_matrix


def calculate_centroid(img, fname=None):
    """
    Calculates the volumetric centroid of a sparsely labeled
    image volume.

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
        List of all centroids in format (label, z, y, x)
    """
    sp_arr = csr_matrix(img.reshape(1, -1))
    uniques = np.unique(sp_arr.data)

    centroids = np.empty((len(uniques), 4))

    for i, label in enumerate(uniques):
        z, y, x = np.unravel_index(
            sp_arr.indices[sp_arr.data == label], img.shape)
        centroids[i] = label, np.mean(z), np.mean(y), np.mean(x)

    if fname:
        np.save(fname, centroids)
    else:
        return centroids

'''
def calculate_centroid(img, label):
    """
    Calculates the volumetric centroid

    Parameters
    ----------
    img : 3d-array like
    label : int
    """
    mask = img == label

    z_bound, y_bound, x_bound = bounding_box(mask)

    mask = mask[z_bound[0]:z_bound[1],
                y_bound[0]:y_bound[1],
                x_bound[0]:x_bound[1]]

    out = np.transpose(np.nonzero(mask))

    z = np.mean(out[:, 0])
    y = np.mean(out[:, 1])
    x = np.mean(out[:, 2])

    centroid_z = z_bound[0] + z
    centroid_y = y_bound[0] + y
    centroid_x = x_bound[0] + x

    return (centroid_z, centroid_y, centroid_x)
'''

def calculate_dimensions(centroid, dimensions, max_dimensions):
    """
    Calculates the coordinates of a box surrounding
    the centroid with shape (z, y, x) dimensions.

    Parameters
    ----------
    centroid : 1d-array like
        Input point with order (z, y, x)
    dimensions : 1d-array like
        Shape of box surrounding centroid with order (z, y, x)
    max_dimensions : 1d-array like
        Maximum dimensions of the data
    """
    z_dim, y_dim, x_dim = dimensions
    z_max, y_max, x_max = max_dimensions

    z_dim = z_dim / 2
    y_dim = y_dim / 2
    x_dim = x_dim / 2

    z_range = np.around([centroid[0] - z_dim if centroid[0] - z_dim > 0 else 0,
                         centroid[0] + z_dim if centroid[0] + z_dim < z_max else z_max]).astype(int)
    y_range = np.around([centroid[1] - y_dim if centroid[1] - y_dim > 0 else 0,
                         centroid[1] + y_dim if centroid[1] + y_dim < y_max else y_max]).astype(int)
    x_range = np.around([centroid[2] - x_dim if centroid[2] - x_dim > 0 else 0,
                         centroid[2] + x_dim if centroid[2] + x_dim < x_max else x_max]).astype(int)

    return z_range, y_range, x_range


def f0(channel, annotation):
    """
    Calculates the integrated intensity given annotation i.

    Parameters
    ----------
    channel : 3d-array like
    annotation : 3d-array like
    """
    synapse = np.sum(np.multiply(channel, annotation > 0))
    around_synapse = np.sum(np.multiply(channel, annotation == 0))

    return synapse, around_synapse


def f1(channel, annotation):
    """
    Calculates the average intensity of a region given annotation i.abs

    Parameters
    ----------
    channel : 3d-array like
    annotation : 3d-array like
    """
    synapse = np.mean(np.multiply(channel, annotation > 0))
    #around_synapse = np.mean(np.multiply(channel, annotation == 0))
    around_synapse = np.mean(channel)

    return synapse, around_synapse


def calculate_feature(centroids, channel, annotation, dimensions, max_dimensions, feature):
    """
    Calculates the

    Parameters
    ----------
    centroids : 1d-array like
    channel : 3d-array like
    annotation : 3d-array like
    dimensions : tuple
    max_dimensions : tuple
    feature : str
        Either 'f0' or f1'
    """

    synapse = []
    around_synapse = []

    for idx, centroid in centroids:
        z, y, x = calculate_dimensions(centroid, dimensions, max_dimensions)
        channel_cutout = channel[z[0]:z[1], y[0]:y[1], x[0]:x[1]]
        annotation_cutout = annotation[z[0]:z[1], y[0]:y[1], x[0]:x[1]] == idx

        if feature == 'f0':
            tmp_synapse, tmp_around = f0(channel_cutout, annotation_cutout)
        elif feature == 'f1':
            tmp_synapse, tmp_around = f1(channel_cutout, annotation_cutout)

        synapse.append(tmp_synapse)
        around_synapse.append(tmp_around)

    return synapse, around_synapse


def get_random_non_synapse(mask, annotation):
    z = np.random.randint(0, 27, 20)
    y = np.random.randint(0, 4518, 20)
    x = np.random.randint(0, 6306, 20)

    non_synapses = []

    for point in itertools.product(z, y, x):
        if mask[point]:
            z_range, y_range, x_range = calculate_dimensions(
                point, 9, 396, 396)
            if np.sum(annotation[z_range[0]:z_range[1],
                                 y_range[0]:y_range[1],
                                 x_range[0]:x_range[1]]) == 0:
                non_synapses.append(point)

    return non_synapses
