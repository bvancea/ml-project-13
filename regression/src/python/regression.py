__author__ = 'bogdan'

import io_utils as io
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

TRAINING_IN = "../resources/training.csv"
VALIDATION_IN = "../resources/validation.csv"
TESTING_IN = "../resources/validation.csv"

TESTING_OUT = './out/testing_y.out'
VALIDATION_OUT = './out/validation_y.out'


#read input data
training_X, training_y, testing_X, validation_X = io.read_regression_data()

#Scale input data with respect to mean and std dev
scaler = StandardScaler()
scaler.fit(training_X)
training_X = scaler.transform(training_X)
testing_X = scaler.transform(testing_X)
validation_X = scaler.transform(validation_X)

rng = np.random.RandomState(1)
#train regressor
#regularizers = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
#regressor = linear_model.RidgeCV(regularizers, normalize=True)
regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=rng)

regressor.fit(training_X, training_y)


#make predictions
testing_y = regressor.predict(testing_X)
validation_y = regressor.predict(validation_X)

#write to output file
io.write_prediction(TESTING_OUT, testing_y)
io.write_prediction(VALIDATION_OUT, validation_y)

