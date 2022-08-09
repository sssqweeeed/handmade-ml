import numpy as np
import pandas as pd


def mse(y_pred, y):
    return np.sum((y - y_pred) ** 2)


def accuracy(y_pred: pd.Series, y: pd.Series):
    y_pred = np.array(y_pred.values)
    y = np.array(y.values)

    return np.where(y_pred == y)[0].size / y_pred.size