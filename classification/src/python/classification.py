__author__ = 'bogdan'

import numpy as np
import io_utils as io
import classification_utils as class_utils

from sklearn import svm, linear_model, grid_search, cross_validation, preprocessing, metrics, ensemble, tree


###############################################################################################
# Read input data
###############################################################################################
training = io.read_x_y(io.TRAINING_IN)
testing_X = io.read_x(io.TESTING_IN)
validation_X = io.read_x(io.VALIDATION_IN)

training_y = training.cls
training_X = training.drop(labels=['cls'], axis=1)


###############################################################################################
# Standardize data
###############################################################################################
print(training_X.shape)
scaler = preprocessing.StandardScaler()
training_X = scaler.fit_transform(training_X)
testing_X = scaler.transform(testing_X)
validation_X = scaler.transform(validation_X)

###############################################################################################
# Train more models and pick the one with the highest R2 score
###############################################################################################

custom_scorer = metrics.make_scorer(class_utils.asymmetric_scorer, greater_is_better=False)
# Least squares model, normalize flag indicates that data should be standardized
# (brought to normal distribution)
models = {}
# params = {"penalty": ["l1"],
#          "C": np.linspace(1.0, 0.001, 10),
#          "class_weight": [{-1: 0.8, 1: 0.2}],
#          "dual": [False]}
# models["logistic_regression"] = grid_search.GridSearchCV(estimator=linear_model.LogisticRegression(),
#                                                         param_grid=params)

# params = {"kernel": ["linear", "rbf", "sigmoid", "poly"],
#           "C": np.linspace(100.0, 0.001, 30),
#           "class_weight": [{-1: 0.8, 1: 0.2}]}
# models["svm"] = grid_search.GridSearchCV(estimator=svm.SVC(),
#                                          param_grid=params,
#                                          scoring=custom_scorer, verbose=3)

params = {"base_estimator": [tree.DecisionTreeClassifier(max_depth=10, min_samples_leaf=10),
							tree.DecisionTreeClassifier(max_depth=10, min_samples_leaf=10, criterion='entropy')],
		  "algorithm": ["SAMME","SAMME.R"],
		  "n_estimators": [200, 500],
		  "learning_rate": [0.05, 0.1, 1]}
models["adaboost"] = grid_search.GridSearchCV(estimator=ensemble.AdaBoostClassifier(), 
											  param_grid=params,
											  scoring=custom_scorer, verbose=3)

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
