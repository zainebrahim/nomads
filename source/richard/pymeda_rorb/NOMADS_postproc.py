import numpy as np
from skimage.measure import label

def postproc(data, thresh):
    label_img = label(data)
    keep_list = []
    for idx in np.unique(label_img):
        if not idx == 0:
            if(np.sum(label_img == idx)) > thresh:
                keep_list.append(idx)
    
    return np.isin(label_img, keep_list)
