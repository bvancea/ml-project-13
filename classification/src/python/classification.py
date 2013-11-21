__author__ = 'bogdan'

import numpy as np
import io_utils as io
import classification_utils as class_utils

from sklearn import svm, tree, linear_model, grid_search, cross_validation, ensemble, preprocessing, metrics, pipeline, feature_selection


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

scaler = preprocessing.StandardScaler()
training_X = scaler.fit_transform(training_X)
testing_X = scaler.transform(testing_X)
validation_X = scaler.transform(validation_X)


###############################################################################################
# Feature engineering
###############################################################################################
#feature_transforms = [np.square]
#feature_adder = class_utils.FeatureAdder(functions=feature_transforms)
#training_X, testing_X, validation_X = map(feature_adder.transform, (training_X, testing_X, validation_X))

print(training_X.shape)

###############################################################################################
# Feature selection
###############################################################################################

feature_selector = feature_selection.SelectPercentile(feature_selection.f_classif)


###############################################################################################
# Train more models and pick the one with the highest R2 score
###############################################################################################

#The scoring function should be our asymmetric score function
custom_scorer = metrics.make_scorer(class_utils.asymmetric_scorer, greater_is_better=False)

# Cross validation model should be a stratified k-fold, meaning that an equal number of positivi
# and negative samples should be chosen

strat_kfold = 10

# Least squares model, normalize flag indicates that data should be standardized
# (brought to normal distribution)
models = {}
C = np.logspace(-1.0, 2.0, 20)
    #C = [1.2, 0.1]
#weights = {-1: 0.8, 1: 0.2}
weights = {-1: 0.8, 1: 0.2}
gamma = [1.5, 1.25, 1.0, 0.75, 0.5, 0.1]

percentiles = [100]
#params = {"estimator__kernel": ["rbf"],
#          "estimator__C": C,
#          "estimator__gamma": gamma,
#          "estimator__class_weight": [weights],
#          "estimator__random_state": [None],
#          "estimator__max_iter": [-1]}
params = {"estimator__kernel": ["rbf"],
          "estimator__C": [4.1],
          "estimator__gamma": [1.25],
          "estimator__class_weight": [weights],
          "estimator__random_state": [None],
          "estimator__max_iter": [10000]}
svm_pipe = pipeline.Pipeline([('anova', feature_selector), ('estimator', svm.SVC())])
for percentile in percentiles:
    svm_pipe.set_params(anova__percentile=percentile)
    models["svm_" + str(percentile)] = grid_search.GridSearchCV(estimator=svm_pipe,
                                                                param_grid=params,
                                                                scoring=custom_scorer,
                                                                verbose=3)


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
estimator = class_utils.MultipleEstimatorCV(test_size=0.5, split_test=False, cv=strat_kfold)
training_X_copy = training_X
training_y_copy = training_y
ranking = estimator.cross_validate_models(models, training_X_copy, training_y_copy)

# Print a ranking of the models
for count, (model, model_name, cv_train_error) in enumerate(ranking):
    print count, model_name, 'Cv Train score : %-10.6f' % cv_train_error

###############################################################################################
# Use the highest ranking model for prediction
###############################################################################################
regr, name, cv_train_error = ranking[0]
# Best classifier
print "Params of best classifier: ", regr.best_params_
print "Score of best classifier: ", (-regr.best_score_)
#fit the whole training data after choosing the model
#regr.fit(training_X, training_y)

validation_y = regr.predict(validation_X)
testing_y = regr.predict(testing_X)

###############################################################################################
# Write to the output file
###############################################################################################
io.write_prediction(io.VALIDATION_OUT, validation_y)
io.write_prediction(io.TESTING_OUT, testing_y)
