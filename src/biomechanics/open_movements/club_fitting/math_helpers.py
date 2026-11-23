import numpy as np
import math
from scipy import signal
from scipy.linalg import svd
from scipy.signal import savgol_filter

def safe_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors along axis 1 while avoiding division by zero."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1e-8
    return vectors / norms

def rms(x: np.ndarray) -> float:
    """Return the root mean square of all values in an array."""
    x = np.asarray(x)
    return np.sqrt(np.mean(x**2))

PITCH_KIN_FC_HZ = 10
