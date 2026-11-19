import numpy as np
import math
import json
from scipy import signal
from scipy.linalg import svd
from scipy.signal import savgol_filter

def safe_normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1e-8
    return vectors / norms

def rms(x):
    x = np.asarray(x)
    return np.sqrt(np.mean(x**2))

PITCH_KIN_FC_HZ = 10
