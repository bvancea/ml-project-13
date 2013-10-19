__author__ = 'bogdan'


import numpy as np
from numpy import genfromtxt
from sklearn import linear_model


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
    return training_ret, training_label, testing_ret,validation_ret


def write_to_result_file(filename, prediction_list):
    prediction_list = np.array(prediction_list)
    prediction_list.tofile(file=filename, sep='\n')

#read input data
training_X, training_y, testing_X, validation_X = read_regression_data()

#train regressor
regressor = linear_model.LinearRegression()
regressor.fit(training_X, training_y)

print regressor.coef_

#make predictions
testing_y = regressor.predict(testing_X)
validation_y = regressor.predict(validation_X)

#write to output file
write_to_result_file('testing_y.txt', testing_y)
write_to_result_file('validation_y.txt', validation_y)

