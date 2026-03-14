'''
Mean Absolute Error Normalised.

Magnitude-invariant universal loss score.
Designed for evaluating the performance of Times Series Forecast models.
Evaluates such problematic cases as Near-Zero and Flat vectors of target values.

Requires min and max reference values.
'''

import logging
import numpy as np


def normalize(
    array: np.array,
    min_: float = None,
    max_: float = None
):
    if len(np.unique(array)) == 1:
        value = np.unique(array)[0]
        norm_value = 0.5
        if max_ != min_:
            norm_value = (value - min_) / (max_ - min_)
        return np.ones_like(array) * norm_value
    if min_ is None:
        min_ = array.min()
    if max_ is None:
        max_ = array.max()
    return (array - min_) / (max_ - min_)


def maen_score(true: np.array, pred: np.array, min_: float, max_: float):
    '''
    min_ and max_: float - reference values
    Usage:\n\n
    y_train: np.array = ...  # target values for training\n
    y_test: np.array = ...  # target values for testing\n
    y_pred: np.array = ...  # y_hat\n\n

    maen = maen_score(y_test, y_pred, y_train.min(), y_train.max())
    '''
    if min_ == max_:
        if not np.abs(true - pred).sum():
            return 0.0
        warning = f'''
        Min -- Max collision: {min_=}, {max_=}.
        If confident that the target values are {min_} only,
        then any other values are infinitely wrong.
        Consider assigning pred = np.ones_like(true) * {min_}.
        '''
        logging.warning(warning)
        return None
    if len(true) != len(pred):
        err_msg = f'Length mismatch: {len(true)=}, {len(pred)=}'
        raise ValueError(err_msg)
    if (len(true.shape) > 1) or (len(pred.shape) > 1):
        err_msg = f'Invalid dimensions: {true.shape=}, {pred.shape=}'
        err_msg += '\nProvide 1-D arrays only.'
        raise ValueError(err_msg)
    true_normalized = normalize(true, min_, max_)
    pred_normalized = normalize(pred, min_, max_)
    maen = np.abs(pred_normalized - true_normalized).mean()
    return maen

