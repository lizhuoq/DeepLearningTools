from package import *

from sklearn.metrics import mean_squared_error, r2_score


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


def rmse(true, pred):
    return mean_squared_error(true, pred, squared=False)


def ubrmse(true, pred):
    return mean_squared_error(true - true.mean(), pred - pred.mean(), squared=False)


def r2(true, pred):
    return r2_score(true, pred)
