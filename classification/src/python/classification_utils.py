__author__ = 'bogdan'
import numpy as np
import pandas as pd
from operator import itemgetter
from sklearn import preprocessing, cross_validation, metrics


class MultipleEstimatorCV:

    def __init__(self, test_set_size):
        self.test_set_size = test_set_size
        self.analyzed_models = []

    def cross_validate_models(self, models, X, y):

        test_size = self.test_set_size
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                                                             test_size=test_size,
                                                                             random_state=42)
        for model_name, model in models.iteritems():
            m = model
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
    for i in range(true_X.size):
        #false positive
        if true_X[i] == 1 and pred_X[i] == -1:
            fp += 1
        #false negative
        if true_X[i] == -1 and pred_X[i] == 1:
            fn += 1
    score = (5 * fp + fn) / float(true_X.size)
    #print("FP: ", fp, " FN: ", fn, " m: ", true_X.size, " score: ", score)
    return score




