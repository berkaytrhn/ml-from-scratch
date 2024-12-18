import numpy as np

def euclidean_distance(v1: np.ndarray, v2: np.ndarray):
    return np.sqrt(np.sum(np.square(v1-v2)))


def manhattan_distance(v1: np.ndarray, v2: np.ndarray):
    return np.sum(np.abs(v1 - v2))