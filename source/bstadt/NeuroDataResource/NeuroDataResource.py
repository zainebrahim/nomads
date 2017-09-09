import pickle
from intern.remote.boss import BossRemote
from intern.resource.boss.resource import ChannelResource

class NeuroDataResource:
    def __init__(self, host, token, collection, experiment, chanList):
        self._collection = collection
        self._experiment = experiment
        self._bossRemote = BossRemote({'protocol':'https',
                                       'host':host,
                                       'token':token})
        self._chanList = {}
        for chan in chanList:
            try:
                self._chanList[str(chan)] = ChannelResource(chan,
                                                            collection,
                                                            experiment,
                                                            'image',
                                                            datatype='uint8')
            except:
                #TODO error handle here
                raise

    def get_cutout(self, chan, zRange, yRange, xRange):
        if not chan in self._chanList.keys():
            print('Error: Channel Not Found in this Resource')
            return
        data = self._bossRemote.get_cutout(self._chanList[chan],
                                           0,
                                           xRange,
                                           yRange,
                                           zRange)
        return data

if __name__ == '__main__':
    host = 'api.boss.neurodata.io'
    token = pickle.load(open('./data/token.pkl', 'rb'))
    myResource = NeuroDataResource(host,
                                  token,
                                  'collman',
                                  'collman15v2',
                                  ['DAPI1st', 'GABA488'])
    cutout = myResource.get_cutout('DAPI1st', [10,20], [10,20], [10,20])
