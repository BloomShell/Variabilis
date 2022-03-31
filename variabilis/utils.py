import numpy as np


def moving_average(x, window= 7, ddof=0):
    return np.convolve(x, np.ones(window), 'valid') / (window - ddof)