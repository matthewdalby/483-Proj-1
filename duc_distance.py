import numpy as np


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def cosine_distance(point1, point2):
    sumyy = (point2 ** 2).sum(1)
    sumxx = (point1 ** 2).sum(1, keepdims=1)
    sumxy = point1.dot(point2.T)
    return (sumxy / np.sqrt(sumxx)) / np.sqrt(sumyy)
