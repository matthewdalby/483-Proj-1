import numpy as np


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def cosine_distance(point1, point2):
    numerator = (point1 * point2).sum()
    denom1 = (point1 * point1).sum()
    denom2 = (point2 * point2).sum()
    return 1 - numerator/np.sqrt(denom1*denom2)
