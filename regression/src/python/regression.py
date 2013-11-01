__author__ = 'bogdan'

import numpy as np

from sklearn import linear_model, grid_search
from pyearth import Earth

import io_utils as io
import regression_utils as reg_utils


###############################################################################################
# Read input data
###############################################################################################
training = io.read_x_y(io.TRAINING_IN)
testing_X = io.read_x(io.TESTING_IN)
validation_X = io.read_x(io.VALIDATION_IN)

###############################################################################################
# perform standardization/normalization of the features, this isn't very useful tbh,
# normalization is done later by each model.
###############################################################################################
standardize = False
normalize = False
training, validation_X, testing_X = reg_utils.preprocess_data(training, validation_X, testing_X,
                                                              normalize=normalize,
                                                              standardize=standardize)

###############################################################################################
# Add some more features according to the correlation matrix between the features and
# the delay.
##############################################################################################
functions = [np.sin, np.log, np.sqrt, np.square]
intra_feature_corr = False
f_adder = reg_utils.FeatureAdder(functions=functions, corr_bound=0.1, intra_corr_func=reg_utils.sum_squares)

train_k = f_adder.kernelize(training, 'delay', intra_feature_corr)
training_y = train_k.delay
training_X = train_k.drop(labels=['delay'], axis=1)
validation_X = f_adder.transform(validation_X, intra_feature_corr)
testing_X = f_adder.transform(testing_X, intra_feature_corr)

# Print the features used
print "Used features", training_X

###############################################################################################
# Train more models and pick the one with the highest R2 score
###############################################################################################

# Least squares model, normalize flag indicates that data should be standardized
# (brought to normal distribution)
models = {}
models["least_squares"] = linear_model.LinearRegression(normalize=True, copy_X=True, verbose=3)

# Regularized models, lasso and ridge
# Alpha values represent the regularization parameter. The best one should be chosen by CV
alphas = np.logspace(-1, 4, 25).tolist()
models['ridge_cv'] = linear_model.RidgeCV(alphas=alphas, cv=10, verbose=3, normalize=True)
models['lasso_cv'] = linear_model.LassoCV(max_iter=100000000, alphas=alphas, cv=10, verbose=3, normalize=True, copy_X=True)

# Open-source MARS implementation, https://github.com/jcrudy/py-earth
# Choose the maximum degree of interaction between Hinge basis functions by CV
earth_params = {'max_degree': range(6)[1:]}
earth_cv = grid_search.GridSearchCV(param_grid=earth_params, estimator=Earth(), cv=10, verbose=3)
models['earth_cv'] = earth_cv

# Train all the models on a fraction of the training data set, test_set_size represents the
# fraction of the training data used only for testing the model performance. Cross-Validation
# for each model is not done on that part of the data
estimator = reg_utils.MultipleEstimatorCV(test_set_size=0.1)
training_X_copy = training_X
training_y_copy = training_y
ranking = estimator.cross_validate_models(models, training_X_copy, training_y_copy)

# Print a ranking of the models
for count, (model, model_name, test_error, train_error) in enumerate(ranking):
    print count, model_name, 'R2 Test score : %-10.6f' % test_error, 'R2 Train score : %-10.6f' % train_error

###############################################################################################
# Use the highest ranking model for prediction
###############################################################################################
regr, name, test_error, train_error = ranking[0]

# Best regressor
print regr.get_params()
#fit the whole training data after choosing the model
#regr.fit(training_X, training_y)
validation_y = regr.predict(validation_X)
testing_y = regr.predict(testing_X)

###############################################################################################
# Write to the output file
###############################################################################################
io.write_prediction(io.VALIDATION_OUT, validation_y)
io.write_prediction(io.TESTING_OUT, testing_y)
