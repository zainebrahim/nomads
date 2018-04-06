import random
import pickle
import numpy as np

def load_and_preproc(filehandle, z_transform=True):
    raw = pickle.load(open(filehandle, 'rb'))
    if z_transform:
        for channel in raw.keys():
            #dont want to z transform annotations
            if channel != 'annotation':
                data = raw[channel]

                #get z transform stats
                for z_idx in range(data.shape[0]):
                    mu = np.mean(data[z_idx])
                    sigma = np.std(data[z_idx])
                    raw[channel][z_idx] = (raw[channel][z_idx] - mu)/sigma

    return raw

def get_train_set(data, r_xy, r_z, channel_list, balance=False, positive_to_negative_ratio=1):

    volume = np.stack([data[channel]\
                       for channel in channel_list])
    #To put channel axis last
    volume = np.moveaxis(volume, 0, -1)


    valid_z_cent = list(range(r_z + 1, volume.shape[0] - r_z - 1))
    valid_y_cent = list(range(r_xy + 1, volume.shape[1] - r_xy - 1))
    valid_x_cent = list(range(r_xy + 1, volume.shape[2] - r_xy - 1))

    training_points = np.array(np.meshgrid(valid_z_cent,
                               valid_y_cent,
                               valid_x_cent)).T.reshape(-1, 3)

    #shuffle input data
    training_points = list(training_points)

    random.shuffle(training_points)

    #NOTE +1 here to center z
    training_examples = [volume[z-r_z:z+r_z+1,
                                y-r_xy:y+r_xy,
                                x-r_xy:x+r_xy,
                                :]\
                         for z, y, x in training_points]

    labels = [[not (data['annotation'][z, y, x] > 0), data['annotation'][z, y, x] > 0]\
              for z, y, x in training_points]

    if balance:
        positive_idxs = [i\
                         for i in range(len(labels))\
                         if labels[i][1]]

        negative_idxs = [i\
                         for i in range(len(labels))\
                         if not labels[i][1]][:int((1/positive_to_negative_ratio)*len(positive_idxs))]

        balanced_idx_list = positive_idxs + negative_idxs

        #This is done to maintain the shuffle
        balanced_idx_idx_list = [i for i in range(len(balanced_idx_list))]
        random.shuffle(balanced_idx_idx_list)


        returnable_examples = [training_examples[balanced_idx_list[i]]\
                               for i in balanced_idx_idx_list]

        returnable_labels = [labels[balanced_idx_list[i]]\
                             for i in balanced_idx_idx_list]

        return returnable_examples, returnable_labels

    return training_examples, labels
