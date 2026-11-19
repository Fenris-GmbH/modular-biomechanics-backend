import numpy as np


def safe_normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1e-8
    return vectors / norms
