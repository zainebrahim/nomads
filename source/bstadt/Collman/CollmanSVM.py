import sys
import pickle
import numpy as np
sys.path.append('../NeuroDataResource')
from skimage.measure import label
from sklearn import svm as sksvm
from NeuroDataResource import NeuroDataResource
from CollmanAlg import CollmanAlg

class CollmanSVM:
    def __init__(self, resource, resolution):
        self._resource = resource

        self._requiredChannels = ['GABA488',
                                  'gephyrin594',
                                  'GAD647',
                                  'Synapsin647',
                                  'GS594',
                                  'VGluT1_647',
                                  'annotation']

        for channel in self._requiredChannels:
            if not self._resource.assert_channel_exists(channel):
                raise ValueError(channel + 'Channel must be in Resource')

        self._medianFilter = CollmanAlg(resource, resolution)
        self._svm = None


    def masked_z_transform(self, mask, channel):
        nonzero = float(np.count_nonzero(mask))
        mu = np.sum(np.multiply(mask, channel))/nonzero
        sigma = np.sqrt(np.sum(np.multiply(mask, (channel-mu)**2))/nonzero)
        return (channel - mu)/float(sigma)


    #TODO this svm example/ feature vector business is hacky. Fix pls
    def make_svm_featureVector(self, clusters, zChannels, clusterIdx):
        mask = np.array(clusters == clusterIdx).astype(int)
        area = float(np.count_nonzero(mask))
        featureVector = [np.sum(np.multiply(mask, channel))/area for channel in zChannels]
        return featureVector


    def make_svm_example(self, clusters, zChannels, labels, clusterIdx):
        mask = np.array(clusters == clusterIdx).astype(int)
        area = float(np.count_nonzero(mask))
        featureVector = [np.sum(np.multiply(mask, channel))/area for channel in zChannels]
        label = np.max(np.multiply(mask, labels)) > 0
        return [label, featureVector]


    def train_on_volume(self, zRange, yRange, xRange):
        clusters = self._medianFilter.detect(zRange, yRange, xRange)
        mask = np.array(clusters > 0).astype(int)
        channelData = [self._resource.get_cutout(channel, zRange, yRange, xRange) \
                       for channel in self._requiredChannels]

        x = channelData[:-1]
        y = channelData[-1]

        zChannels = [self.masked_z_transform(mask, channel) for channel in x]

        examples = [self.make_svm_example(clusters, zChannels, y, clusterIdx)\
                    for clusterIdx in range(1, np.max(clusters) + 1)]


        self._svm = sksvm.SVC()
        self._svm.fit([elem[1] for elem in examples], [elem[0] for elem in examples])
        return


    def predict_on_volume(self, zRange, yRange, xRange):


        clusters = self._medianFilter.detect(zRange, yRange, xRange)
        mask = np.array(clusters > 0).astype(int)

        x = [self._resource.get_cutout(channel, zRange, yRange, xRange) \
             for channel in self._requiredChannels\
             if not channel == 'annotation']

        zChannels = [self.masked_z_transform(mask, channel) for channel in x]

        examples = [self.make_svm_featureVector(clusters, zChannels, clusterIdx)\
                    for clusterIdx in range(1, np.max(clusters) + 1)]

        #idx + 1 here since examples do not include background cluster
        predictions = [idx + 1 for idx, example in enumerate(examples)\
                      if self._svm.predict(np.array(example).reshape(1, -1))]

        return np.isin(clusters, predictions)
