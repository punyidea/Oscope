import numpy as np

def normalize(np_array):
    return np_array/(np.abs(np_array).max())