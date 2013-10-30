__author__ = 'bogdan'

from sklearn import linear_model, feature_selection, preprocessing, cross_validation, metrics, svm
from pyearth import Earth

import io_utils as io
import regression_utils as reg_utils
import numpy as np

#read input data
training = io.read_x_y(io.TRAINING_IN)
testing_X = io.read_x(io.TESTING_IN)
validation_X = io.read_x(io.VALIDATION_IN)

#perform standardization/normalization of the features
standardize = True
normalize = False
training, validation_X, testing_X = reg_utils.preprocess_data(training, validation_X, testing_X,
                                                              normalize=normalize,
                                                              standardize=standardize)

#add some more features
functions = [np.sin, np.square, np.sqrt]
intra_feature_corr = True
f_adder = reg_utils.FeatureAdder(functions=functions, corr_bound=0.1, intra_corr_func=reg_utils.sum_squares)

train_k = f_adder.kernelize(training, 'delay', intra_feature_corr)
training_y = train_k.delay
training_X = train_k.drop(labels=['delay'], axis=1)
validation_X = f_adder.transform(validation_X, intra_feature_corr)
testing_X = f_adder.transform(testing_X, intra_feature_corr)

print training_X
training_X_train, training_X_test, training_y_train, training_y_test = \
    cross_validation.train_test_split(training_X, training_y, train_size=0.65)

alphas = np.logspace(-10, -2, 10).tolist()
reg = linear_model.RidgeCV(alphas=alphas)

#perform feature selection to eliminate random features
#training, validation_X, testing_X = reg_utils.do_feature_selection(regr, training, validation_X, testing_X)

regr = Earth(max_degree=3)
regr.fit_transform(np.array(training_X), np.array(training_y))
validation_y = regr.predict(np.array(validation_X))
testing_y = regr.predict(np.array(testing_X))

#write to output file
io.write_prediction(io.VALIDATION_OUT, validation_y)
io.write_prediction(io.TESTING_OUT, testing_y)
