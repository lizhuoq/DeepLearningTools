from package import *


def custom_pearson(x, y):
    x_mean = x.mean()
    y_mean = y.mean()
    x_minus_mean = (x - x_mean)
    y_minus_mean = (y - y_mean)
    x_minus_mean_square = x_minus_mean ** 2
    y_minus_mean_square = y_minus_mean ** 2
    return (x_minus_mean * y_minus_mean).sum() / np.sqrt(x_minus_mean_square.sum() * y_minus_mean_square.sum())


def numpy_pearson(x, y):
    return np.corrcoef(x, y)[0][1]


