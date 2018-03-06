import numpy as np
from skimage import io

def load_file(filepath):
    return np.array(io.imread(filepath)).astype(np.uint8)

def dump_file(filepath, data):
    io.imsave(filepath, data.astype(np.uint8))
    return
