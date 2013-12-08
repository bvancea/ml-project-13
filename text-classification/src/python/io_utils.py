__author__ = 'bogdan'

import numpy as np
import pandas as pd

TRAINING_IN = "../resources/training.csv"
VALIDATION_IN = "../resources/validation.csv"
TESTING_IN = "../resources/testing.csv"
TESTING_OUT = './out/testing_y.out'
VALIDATION_OUT = './out/validation_y.out'

#some relevant column names
headers = ["name", "city", "country"]


def read_x(filename=None, header_names=None):
    if header_names is None:
        header_names = ["name"]

    return pd.read_csv(filename, header=None, names=header_names, sep=',')


def read_x_y(filename=None, header_names=None):
    if header_names is None:
        header_names = headers

    x_y = pd.read_csv(filename, header=None, names=header_names)

    return x_y


def write_prediction(filename, prediction_list):
    np.savetxt(VALIDATION_OUT, prediction_list, fmt="%d,%d")