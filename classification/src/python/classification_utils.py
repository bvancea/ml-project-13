__author__ = 'bogdan'
import numpy as np
import pandas as pd
from operator import itemgetter
from sklearn import preprocessing, cross_validation, metrics
from itertools import izip


class MultipleEstimatorCV:

    def __init__(self, test_size, split_test, cv):
        self.test_size = test_size
        self.split_test = split_test
        self.internal_cv_fold = cv
        self.analyzed_models = []

    def cross_validate_models(self, models, X, y):
        X_test = None
        y_test = None
        X_train = None
        y_train = None
        internal_cv = None
        if self.split_test:
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=self.test_size)
            internal_cv = cross_validation.StratifiedKFold(y_train, n_folds=self.internal_cv_fold)
        else:
            internal_cv = cross_validation.StratifiedKFold(y, n_folds=self.internal_cv_fold)
            X_train = X
            y_train = y

        for model_name, model in models.iteritems():
            m = model
            m.set_params(cv=internal_cv)
            m.fit(X_train, y_train)

            for params, mean_score, scores in sorted(m.grid_scores_, key=itemgetter(1)):
                print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))

            if X_test is not None:
                y_pred = m.predict(X_test)
                score = asymmetric_scorer(y_test, y_pred)
                print "Validation score: ", score

            #error = metrics.mean_squared_error(y_test, y_pred)
            self.analyzed_models.append((m, model_name, (-m.best_score_)))

        return sorted(self.analyzed_models, key=itemgetter(2))


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


def cube(x):
    return np.power(x, 3)


def quad(x):
    return np.power(x, 4)


class FeatureAdder:

    def __init__(self, functions):
        self.functions = functions

    def transform(self, X):

        new_aggreg_X = X
        for func in self.functions:
            new_X = func(X)
            new_aggreg_X = np.hstack((new_aggreg_X, new_X))

        return new_aggreg_X




