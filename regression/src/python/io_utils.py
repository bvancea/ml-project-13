import numpy as np
import pandas as pd

from numpy import genfromtxt

TRAINING_IN = "../resources/training.csv"
VALIDATION_IN = "../resources/validation.csv"
TESTING_IN = "../resources/testing.csv"
TESTING_OUT = './out/testing_y.out'
VALIDATION_OUT = './out/validation_y.out'


#some relevant column names
headers = ['width','rob','iq','lsq','rfsize','rfread','rfwrite','gshare','btb','branches','l1icache','l1dcache','l2ucache','depth','delay']


def read_x(filename=None, header_names=None):
    if header_names is None:
        header_names = headers[:14]

    return pd.read_csv(filename, header=None, names=header_names)


def read_x_y(filename=None, header_names=None):
    if header_names is None:
        header_names = headers

    x_y = pd.read_csv(filename, header=None, names=header_names)

    return x_y

def write_prediction(filename, prediction_list):
    prediction_list = np.array(prediction_list)
    prediction_list.tofile(file=filename, sep='\n')