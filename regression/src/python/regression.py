from cssutils.helper import string

__author__ = 'bogdan'

from sklearn import linear_model, feature_selection, preprocessing, cross_validation, metrics, svm, isotonic
from pyearth import Earth

import io_utils as io
import regression_utils as reg_utils
import numpy as np

#read input data
training = io.read_x_y(io.TRAINING_IN)
testing_X = io.read_x(io.TESTING_IN)
validation_X = io.read_x(io.VALIDATION_IN)

#perform standardization/normalization of the features, this isn't very useful tbh
standardize = False
normalize = False
training, validation_X, testing_X = reg_utils.preprocess_data(training, validation_X, testing_X,
                                                              normalize=normalize,
                                                              standardize=standardize)

#add some more features
functions = [np.sin, np.log, np.sqrt, np.square, reg_utils.cube, reg_utils.quad]
intra_feature_corr = True
f_adder = reg_utils.FeatureAdder(functions=functions, corr_bound=0.15, intra_corr_func=reg_utils.sum_squares)

train_k = f_adder.kernelize(training, 'delay', intra_feature_corr)
training_y = train_k.delay
training_X = train_k.drop(labels=['delay'], axis=1)
validation_X = f_adder.transform(validation_X, intra_feature_corr)
testing_X = f_adder.transform(testing_X, intra_feature_corr)

print "Used features", training_X
#initialize with the dumbest models
models = {}

#lets also try ridge
alphas = np.logspace(-1, 4, 25).tolist()
#models["least_squares"] = linear_model.LinearRegression(normalize=True, copy_X=True)
models['ridge_cv'] = linear_model.RidgeCV(alphas=alphas, cv=4, normalize=True)
models['lasso_cv'] = linear_model.LassoCV(alphas=alphas, cv=4, normalize=True)

copy_X = True

for alpha in alphas:
    label = "Ridge Regression using alpha: " + '%-15.6s' % `alpha`
    models[label] = linear_model.Ridge(alpha=alpha, normalize=True, copy_X=copy_X)

label = "Bayesian Ridge Regression "
models[label] = linear_model.BayesianRidge(normalize=True, copy_X=copy_X)

label = "RBF SVR Regression "
models[label] = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)

for alpha in alphas:
    label = "Lasso Regression using alpha: " + '%-10.6s' % `alpha`
    models[label] = linear_model.Lasso(alpha=alpha, max_iter=100000000, normalize=True, copy_X=copy_X)

for alpha in alphas:
    label = "Multi-task Lasso using alpha: " + '%-15.6s' % `alpha`
    models[label] = linear_model.MultiTaskLasso(alpha, normalize=True, copy_X=True)

for alpha in alphas:
    label = "Elastic net using alpha: " + '%-15.6s' % `alpha`
    models[label] = linear_model.ElasticNet(alpha, normalize=True, copy_X=True)

for alpha in alphas:
    label = "Multi-task Elastic net using alpha: " + '%-15.6s' % `alpha`
    models[label] = linear_model.MultiTaskElasticNet(alpha, normalize=True, copy_X=True)

#try earth
for degree in range(6)[1:]:
    label = "Earth degree " + `degree`
    models[label] = Earth(max_degree=degree)

estimator = reg_utils.MultipleEstimatorCV(k=10)
training_X_copy = training_X
training_y_copy = training_y
ranking = estimator.cross_validate_models(models, training_X_copy, training_y_copy)

for count, (model, model_name, test_error, train_error) in enumerate(ranking):
    print count, model_name, 'Test error : %-10.6f' % test_error, 'Train error : %-10.6f' % train_error

regr, name, test_error, train_error = ranking[0]

#fit whatever
regr.fit(training_X, training_y)
validation_y = regr.predict(validation_X)
testing_y = regr.predict(testing_X)

#write to output file
io.write_prediction(io.VALIDATION_OUT, validation_y)
io.write_prediction(io.TESTING_OUT, testing_y)
