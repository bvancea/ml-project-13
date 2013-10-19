import numpy as np
from numpy import genfromtxt


def read_regression_data(training_file=None,testing_file=None,validation_file=None, delimiter=None):
    if delimiter is None:
        delimiter = ','
    if training_file is None:
        training_file = '../resources/training.csv'

    training_ret = genfromtxt(training_file, delimiter=delimiter, usecols=range(13))
    training_label = genfromtxt(training_file, delimiter=delimiter, usecols=14)
    if validation_file is None:
        validation_file = '../resources/validation.csv'
    validation_ret = genfromtxt(validation_file, delimiter=delimiter, usecols=range(13))
    if testing_file is None:
        testing_file = '../resources/testing.csv'
    testing_ret = genfromtxt(testing_file, delimiter=delimiter, usecols=range(13))
    return training_ret, training_label, testing_ret, validation_ret


def write_prediction(filename, prediction_list):
    prediction_list = np.array(prediction_list)
    prediction_list.tofile(file=filename, sep='\n')