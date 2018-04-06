import pickle
import numpy as np
from tqdm import tqdm
from skimage.color import gray2rgb
from NOMADS_postproc import postproc
from Quality import make_overlap_img

from NOMADS_beta import NomadsBeta
from data_handler import load_and_preproc, get_train_set

data = load_and_preproc('./data/rorb_data.data', True)
biomarker_list = ['PSD95', 'synapsin', 'Gephyrin', 'VGlut1', 'GABA', 'GAD2', 'GluN1']
train_data = {}
test_data = {}
for key in biomarker_list + ['annotation']:
    vol = data[key]
    train_data[key] = vol[vol.shape[0]//2:, :, :]
    test_data[key] = vol[:vol.shape[0]//2, :, :]

train_features, train_labels = get_train_set(train_data, 8, 1, biomarker_list, balance=True, positive_to_negative_ratio=1/1.5)
test_features, test_labels = get_train_set(test_data, 8, 1, biomarker_list, balance=True, positive_to_negative_ratio=1/1.5)

model = NomadsBeta(len(biomarker_list),
                   learning_rate = 1e-4,
                   decay=0.005)

batch_size = 512
epochs = 5
losses = []
f1s = []

for epoch in range(epochs):
    #50 Iterations for sake of Demo
    for iteration, batch_start_idx in enumerate(range(0, len(train_features), batch_size)):
        batch_features = np.stack(train_features[batch_start_idx:batch_start_idx+batch_size])
        batch_labels = np.stack(train_labels[batch_start_idx:batch_start_idx+batch_size])
        if not iteration % 2:
            batch_pretrain_pred = model.predict_on_batch(batch_features)
            tp = 0
            fp = 0
            fn = 0
            for i in range(len(batch_labels)):
                if np.argmax(batch_pretrain_pred[i]) == np.argmax(batch_labels[i]):
                    if np.argmax(batch_labels[i]):
                        tp +=1
                else:
                    if np.argmax(batch_labels[i]):
                        fn +=1
                    else:
                        fp +=1

            prec = tp/(tp+fp+1)
            rec = tp/(tp+fn+1)
            f1 = 2*prec*rec/(prec+rec+1)
            f1s.append(f1)

        cur_loss = model.train_on_batch(batch_features, batch_labels)
        losses.append(cur_loss)

model.checkpoint("models/model.h5")
