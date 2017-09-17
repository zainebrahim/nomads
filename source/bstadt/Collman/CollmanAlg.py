import sys
import pickle
import numpy as np
sys.path.append('../NeuroDataResource')
from skimage.measure import label
from NeuroDataResource import NeuroDataResource

#NOTE all calculations done in nanometers!!

class CollmanAlg:
    def __init__(self, resource, resolution):
        self._resource = resource
        self._resolution = resolution #resoltuion is nm per pixel
        if not self._resource.assert_channel_exists('PSD95_488'):
            print('ERROR: PSD95_488 Channel must be in Resource')


    def get_cluster_median(self, data):
        maxCluster = np.max(data)
        areas = np.bincount(data.astype(int).flatten())
        return np.median(areas[1:]) #dont count background


    def get_cutoff(self, data, myMin, myMax, target):
        #NOTE target is in nm^2
        if myMin == myMax:
            return myMin

        pivot = (myMin + myMax)/2
        pixelMedian = self.get_cluster_median(np.array(label(data > pivot).astype(int)))
        nmMedian = pixelMedian * self._resolution**2
        if nmMedian < target:
            return self.get_cutoff(data, pivot, myMax, target)

        elif nmMedian > target:
            return self.get_cutoff(data, myMin, pivot, target)

        else:
            return pivot


    def detect(self, zRange, yRange, xRange):
        data = self._resource.get_cutout('PSD95_488',
                                          zRange,
                                          yRange,
                                          xRange)


        unmerged = np.stack([(elem > self.get_cutoff(elem, 0, np.max(data), 9e4)).astype(int)\
                    for elem in data])

        merged = label(unmerged) #TODO better merging here
        return merged



if __name__ == '__main__':
        myToken = pickle.load(open('../NeuroDataResource/data/token.pkl', 'rb'))

        ndr = NeuroDataResource('api.boss.neurodata.io',
                                myToken,
                                'collman',
                                'collman15v2',
                                [{'name': 'PSD95_488', 'dtype':'uint8'}])

        alg = CollmanAlg(ndr, 3.)
        alg.detect([4, 14], [0, 1000], [0, 1000])
