import numpy as np
import pandas as pd
import sys
sys.path.append('../../bstadt/NeuroDataResource')
from functools import partial
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 
from NeuroDataResource import NeuroDataResource
from scipy.sparse import csr_matrix


class Synaptome:
    """
    TODO: Write docs. Also open for suggestions for class name. 
    """
    def __init__(self, host, token, collection, experiment):
        """
        Constructor

        Parameters
        ----------
        host : str
        token : str
            Boss API key
        collection : str
            Name of collection in Boss
        experiment : str
            Name of experiment within collection
        """
        self._resource = NeuroDataResource(host, token, collection, experiment)
        self.channels = self._resource.channels
        self.max_dimensions = self._resource.max_dimensions
        self.voxel_size = self._resource.voxel_size
        self.labels = None
        self.centroids = None


    def F0(self, channel_list, size, mask=None):
        """
        Calculates integrated sum for given box size built around each centroids

        Parameters
        ----------
        channel_list : list of str
            List of channels to calculate F0
        size : 1d-array like
            Size of boxes to build around centroid in (z, y, x) format.
        mask : optional
        """
        assert np.any(self.centroids) != None, "Please run calculate_centroids method."

        dimensions = self._calculate_dimensions(self.centroids, size, self.max_dimensions)
        data = np.empty((len(dimensions), len(channel_list)), dtype=np.uint64)

        for i, channel in enumerate(channel_list):
            print('Calculating features on {}'.format(channel))
            with ThreadPool(processes=8) as tp: #optimum number of connections is 8
                func = partial(self._resource.get_cutout, channel)
                results = tp.starmap(func, dimensions)
                data[:, i] = np.array(list(map(np.sum, results)))

        return pd.DataFrame(data, index=self.labels, columns=channel_list)


    def calculate_centroids(self, annotation_channel):
        """
        Calculates the volumetric centroid of a sparsely labeled
        image volume.

        Parameters
        ----------
        annotation_channel : str
            Name of sparsely labeled image channel

        Returns
        -------
        labels : ndarray
            List of all
        centroids : ndarray
            List of all centroids in format (z, y, x).
        """

        print("Downloading {} channel".format(annotation_channel))
        img = self._resource.get_cutout(annotation_channel, 
                                        [0, self.max_dimensions[0]], 
                                        [0, self.max_dimensions[1]], 
                                        [0, self.max_dimensions[2]])

        print("Calculating centroids")
        sp_arr = csr_matrix(img.reshape(1, -1))
        uniques = np.unique(sp_arr.data)

        centroids = np.empty((len(uniques), 3))

        for i, label in enumerate(uniques):
            z, y, x = np.unravel_index(
                sp_arr.indices[sp_arr.data == label], img.shape)

            centroids[i] = np.mean(z), np.mean(y), np.mean(x)

        self.centroids = centroids
        self.labels = uniques


    def _calculate_dimensions(self, centroids, size, max_dimensions):
        """
        Calculates the coordinates of a box surrounding
        the centroid with size (z, y, x).

        Parameters
        ----------
        centroids : 2d-array like
            Input point with order (z, y, x)
        size : 1d-array like
            Shape of box surrounding centroid with order (z, y, x)
        max_dimensions : 1d-array like
            Maximum dimensions of the data with order (z, y, x)
        """
        z_dim, y_dim, x_dim = [(i - 1) // 2 for i in size]
        z_max, y_max, x_max = max_dimensions

        grid = np.array([[-z_dim, z_dim],
                         [-y_dim, y_dim],
                         [-x_dim, x_dim]])

        out = np.empty((len(centroids), 3, 2), dtype=np.int)

        for i, centroid in enumerate(centroids.astype(np.int)):
            out[i, :, :] = grid + centroid.reshape((-1, 1))

        np.clip(out[:, 0, :], a_min=0, a_max=z_max, out=out[:, 0, :])
        np.clip(out[:, 1, :], a_min=0, a_max=y_max, out=out[:, 1, :])
        np.clip(out[:, 2, :], a_min=0, a_max=x_max, out=out[:, 2, :])

        return out
