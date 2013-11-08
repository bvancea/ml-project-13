__author__ = 'bogdan'

import numpy as np
import io_utils as io
import classification_utils as class_utils

from sklearn import svm, linear_model, grid_search, cross_validation


###############################################################################################
# Read input data
###############################################################################################
training = io.read_x_y(io.TRAINING_IN)
testing_X = io.read_x(io.TESTING_IN)
validation_X = io.read_x(io.VALIDATION_IN)

training_y = training.cls
training_X = training.drop(labels=['cls'], axis=1)
###############################################################################################
# Train more models and pick the one with the highest R2 score
###############################################################################################

# Least squares model, normalize flag indicates that data should be standardized
# (brought to normal distribution)
models = {}
params = {"penalty": ["l2"],
          "C": np.linspace(1.0, 0.001, 10),
          "class_weight": [{-1: 0.8, 1: 0.2}],
          "fit_intercept": [False, True],
          "dual": [False, True]}
models["logistic_regression"] = grid_search.GridSearchCV(estimator=linear_model.LogisticRegression(),
                                                         param_grid=params,
                                                         cv=10)

params = {"kernel": ["linear", "rbf", "sigmoid", "poly"],
          "C": np.linspace(1.0, 0.001, 10),
          "class_weight": [{-1: 0.8, 1: 0.2}]}
models["svm"] = grid_search.GridSearchCV(estimator=svm.SVC(),
                                         param_grid=params)
# Train all the models on a fraction of the training data set, test_set_size represents the
# fraction of the training data used only for testing the model performance. Cross-Validation
# for each model is not done on that part of the data
estimator = class_utils.MultipleEstimatorCV(test_set_size=0.1)
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
