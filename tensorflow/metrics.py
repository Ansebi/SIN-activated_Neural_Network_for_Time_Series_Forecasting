import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import logging


def get_adjusted_mape_bias(reference: np.array, epsilon=1e-5):
    amape_bias = np.mean(reference - np.min(reference)) - np.min(reference)
    n_zeros = np.count_nonzero(np.abs(reference + amape_bias) < epsilon)
    if n_zeros:
        warning = f'Flat input, found {n_zeros=} with {epsilon=}.'
        warning += '\nTo continue comparing against a flat value AMAPE bias is set to zero.'
        logging.warning(warning)
        return 0
    return amape_bias


def mape_score(y_true, y_pred):
    return 100 * mean_absolute_percentage_error(y_true, y_pred)


def amape_score(
        y_true: np.array,
        y_pred: np.array,
        reference: np.array = None
):
    """
    reference: np.array, recommended usage is passing y_train
    as the values known to us by default, therefore forming the expectations
    for handling the data.
    """
    if reference is None:
        warning = f'Unless you are absolutely sure, consider passing the reference.'
        warning += '\nAs the pull of expected values, y_train is usually the one to use.'
        warning += '\nDue to the absence of reference, y_true is now being assigned instead.'
        logging.warning(warning)
        reference = y_true

    amape_bias = get_adjusted_mape_bias(reference)
    try:
        amape = mean_absolute_percentage_error(
            y_true + amape_bias,
            y_pred + amape_bias
        ) * 100
    except TypeError:
        amape = mean_absolute_percentage_error(
            np.array(y_true) + amape_bias,
            np.array(y_pred) + amape_bias
        ) * 100
    return amape


# def default_amape_score(
#         y_true: np.array,
#         default_prediction: np.array = None,
#         y_train: np.array = None,
# ):
#     """
#     The model has to give results more accurate than taking the average out of data.
#     This idea is regarded as common sense below.
#     Either default_prediction or y_train must be provided
    
#     default_prediction (np.array): if no default-level prediction given,
#     it is replaced by the mean of y_train.
#     """
#     if y_train is None:
#         y_train = y_true
    
#     if default_prediction is None:
#         default_prediction = np.ones(len(y_true)) * np.mean(y_train)
    
#     common_sense_amape_bar = amape_score(y_true, default_prediction, reference=y_train)
#     return common_sense_amape_bar


# def default_mape_score(
#         y_true: np.array,
#         default_prediction: np.array = None,
#         y_train: np.array = None,
# ):
#     """
#     The model has to give results more accurate than taking the average out of data.
#     This idea is regarded as common sense below.
#     Either default_prediction or y_train must be provided
    
#     default_prediction (np.array): if no default-level prediction given,
#     it is replaced by the mean of y_train.
#     """    
#     if default_prediction is None:
#         if y_train is None:
#             err_msg = 'If no default predictions provided,'
#             err_msg += 'must provide the refference data for the mean'
#             raise Exception(err_msg)
#         default_prediction = np.ones(len(y_true)) * np.mean(y_train)
    
#     common_sense_mape_bar = mape_score(y_true, default_prediction)
#     return common_sense_mape_bar


def get_dummy_average_prediction(
        y_train: np.array,
        y_true: np.array        
        ):
    dummy_average_prediction = np.ones(len(y_true)) * np.mean(y_train)
    return dummy_average_prediction


def a_score(
        amape,
        default_amape
        ):
    """
    Is 100% if model's AMAPE (Adjusted Mean Absolute Percenage Error) equals zero.
    0% means that the model works no better than calculating the average from the data
    Negative numbers are possible and they mean that the model is worse tahn it's absence
    (i.e. simply predicting the average from the history of the data points)
    """
    amape_delta = default_amape - amape
    a_score = 100 * amape_delta / default_amape
    return a_score


def aim_score(
        accuracy,
        default_accuracy
        ):
    """
    Accuracy improvement, absolute value
    """
    return accuracy - default_accuracy


def raim_score(
        accuracy,
        default_accuracy
        ):
    """
    Relative accuracy improvement
    """
    raim = 100 * (accuracy - default_accuracy) / default_accuracy
    return raim