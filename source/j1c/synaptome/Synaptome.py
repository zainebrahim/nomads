import numpy as np
import pandas as pd
import sys
sys.path.append('../../bstadt/NeuroDataResource')
from NeuroDataResource import NeuroDataResource
from scipy.sparse import csr_matrix
from tqdm import tqdm_notebook
from time import sleep

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

        data = np.empty((len(dimensions), len(channel_list)))

        for i, channel in enumerate(channel_list):
            for j, dimension in tqdm_notebook(enumerate(dimensions), total=len(data), desc=channel):
                z, y, x = dimension

                img = self._resource.get_cutout(channel, z, y, x)

                data[j, i] = np.sum(img)
                sleep(0.01)
        
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
        print("Finished downloading {}".format(annotation_channel))

        print("Calculating centroids")
        sp_arr = csr_matrix(img.reshape(1, -1))
        uniques = np.unique(sp_arr.data)

        labels = np.empty(len(uniques))
        centroids = np.empty((len(uniques), 3))

        for i, label in enumerate(uniques):
            z, y, x = np.unravel_index(
                sp_arr.indices[sp_arr.data == label], img.shape)

            centroids[i] = np.mean(z), np.mean(y), np.mean(x)
            labels[i] = label
        print("Finished calculating centroids")

        self.centroids = centroids
        self.labels = labels


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
