import itertools
import numpy as np
from skimage.measure import block_reduce


def calculate_centroid(arr, label):
    """
    Calculates the volumetric centroid

    Parameters
    ----------
    arr : 3d-array like
    label : int
    """
    out = np.transpose(np.nonzero(arr == label))

    z = np.mean(out[:, 0])
    y = np.mean(out[:, 1])
    x = np.mean(out[:, 2])

    return (z, y, x)


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


def f1_inverse(annotation, channel, block_size, reduced=True):
    a = np.mgrid[-5:6, -5:6]
    distance_matrix = np.sqrt(np.add(np.square(a[0]), np.square(a[1])))

    if reduced:
        synapse_sum = 0
        around_synapse_sum = 0

        annotation = block_reduce(annotation, block_size, np.mean)
        channel = block_reduce(channel, block_size, np.mean)
        
        for arr in channel:
            synapse_sum += np.sum(np.multiply(np.multiply(annotation > 0, arr), distance_matrix))
            around_synapse_sum += np.sum(np.multiply(np.multiply(annotation == 0, arr), distance_matrix))

    return synapse_sum, around_synapse_sum


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
