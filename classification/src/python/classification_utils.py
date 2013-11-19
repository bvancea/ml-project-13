__author__ = 'bogdan'
import numpy as np
import pandas as pd
from operator import itemgetter
from sklearn import preprocessing, cross_validation, metrics
from itertools import izip


class MultipleEstimatorCV:

    def __init__(self, test_set_size, cv):
        self.test_set_size = test_set_size
        self.internal_cv_fold = cv
        self.analyzed_models = []

    def cross_validate_models(self, models, X, y):

        test_size = self.test_set_size
        kfold = cross_validation.StratifiedKFold(y, n_folds=1/test_size)
        (train_index, test_index) = [(train, test) for (train, test) in kfold][0]
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        internal_cv = cross_validation.StratifiedKFold(y_train, n_folds=self.internal_cv_fold)

        for model_name, model in models.iteritems():
            m = model
            m.set_params(cv=internal_cv)
            m.fit(X_train, y_train)
            predicted_y_test = m.predict(X_test)
            predicted_y_train = m.predict(X_train)

            train_error = asymmetric_scorer(y_train, predicted_y_train)
            test_error = asymmetric_scorer(y_test, predicted_y_test)

            #error = metrics.mean_squared_error(y_test, y_pred)
            self.analyzed_models.append((m, model_name, test_error, train_error))

        return sorted(self.analyzed_models, key=itemgetter(2, 3))


def asymmetric_scorer(true_X, pred_X):

    fp = 0
    fn = 0
    for true_Xi, pred_Xi in izip(true_X, pred_X):
        #false positive
        if true_Xi == 1 and pred_Xi == -1:
            fp += 1
        #false negative
        if true_Xi == -1 and pred_Xi == 1:
            fn += 1
    score = (5 * fp + fn) / float(true_X.size)
    #print("FP: ", fp, " FN: ", fn, " m: ", true_X.size, " score: ", score)
    return score




