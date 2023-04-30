import numpy as np

def mse(y_true, y_pred):
    """ Mean squared error"""
    return np.mean(np.square(y_true - y_pred))

def mae(y_true, y_pred):
    """ Mean absolute error"""
    return np.mean(np.abs(y_true - y_pred))

def smape(y_true, y_pred):
    """ Symmetric mean absolute percentage error"""
    return np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))