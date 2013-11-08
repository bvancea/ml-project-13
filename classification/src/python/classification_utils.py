__author__ = 'bogdan'
import numpy as np
import pandas as pd
from operator import itemgetter
from sklearn import preprocessing, cross_validation


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
            train_error = m.score(X_train, y_train)
            test_error = m.score(X_test, y_test)
            #error = metrics.mean_squared_error(y_test, y_pred)
            self.analyzed_models.append((m, model_name, test_error, train_error))

        return sorted(self.analyzed_models, key=itemgetter(2, 3), reverse=True)