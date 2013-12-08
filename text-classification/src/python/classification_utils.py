from itertools import izip

__author__ = 'bogdan'
import numpy as np
from sklearn import feature_extraction, metrics


class HtraeClassifierCV:

    def __init__(self,  country_classifier, city_classifier):
        self.country_classifier = country_classifier
        self.city_classifier = city_classifier

    def fit(self,):
        self.country_codes = []

    def fit(self, X):
        X_name = self.extract_name(X)
        X_city = self.extract_city(X)
        X_country = self.extract_country(X)
        self.country_classifier.fit(X_name, X_country)
        self.city_classifier.fit(X)

    def predict(self, X):
        pass

    def extract_city(self, X):
        return X['city']

    def extract_country(self, X):
        return X['country']

    def extract_name(self, X):
        return X['name']

    def extract_name_features(self, X_name):
        pass

    def country_matrix(country_code):
        a = [1 if code == country_code else 0 for code in country_codes]
        return np.array(a)

def city_score(true_X, pred_X):

    penalty = 0
    for true_Xi, pred_Xi in izip(true_X, pred_X):
        if (true_Xi != true_Xi):
            penalty += 1

    return penalty

def country_score(true_X, pred_X):

    penalty = 0
    for true_Xi, pred_Xi in izip(true_X, pred_X):
        if (true_Xi != true_Xi):
            penalty += 1

    return penalty

