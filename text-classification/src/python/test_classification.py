__author__ = 'bogdan'

import numpy as np
import io_utils as io
import classification_utils as utils

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer

import io_utils as io
import pandas as pd




###############################################################################################
# Read input data
###############################################################################################
training = io.read_x_y(io.TRAINING_IN)
validation = io.read_x(io.VALIDATION_IN, ["name"])
test = io.read_x(io.TESTING_IN, ["name"])

###############################################################################################
# Process city name data
###############################################################################################
vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(2, 2))
X_name_train = vectorizer.fit_transform(training["name"])
X_name_valid = vectorizer.transform(validation["name"])
X_name_test = vectorizer.transform(test["name"])

###############################################################################################
# Make some custom scorers
###############################################################################################
country_scorer = make_scorer(utils.city_score, greater_is_better=False)
city_scorer = make_scorer(utils.country_score, greater_is_better=False)

###############################################################################################
# Train a Multinomial Naive Bayes
###############################################################################################

params = {"alpha": [0.0, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]}
mnb_country = GridSearchCV(estimator=MultinomialNB(), param_grid=params, cv=5)
#First find the countries
mnb_country.fit(X_name_train.toarray(), training["country"])
y_country = mnb_country.predict(X_name_valid.toarray())

country_codes = list(set(training["country"]))
params = {"alpha": [0.0, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]}

mnb_city = GridSearchCV(estimator=MultinomialNB(), param_grid=params, cv=5)
#ucc = np.frompyfunc(country_matrix, 1, 1)
#extended_city_train = np.hstack((X_name_train.toarray(), np.array(ucc(training["country"]))))
extended_city_train = X_name_train.toarray()
mnb_city.fit(extended_city_train, training["city"])

X_valid_array = X_name_valid.toarray()
#extended_city_valid = np.hstack((X_name_valid.toarray, np.array(ucc(y_country))))
extended_city_valid = X_valid_array
y_city = mnb_city.predict(extended_city_valid)

print("Best params for country predictor: ", mnb_country.best_params_)
print("Best score for country ", mnb_country.best_score_)

print("Best params for city predictor: ", mnb_city.best_params_)
print("Best score for city ", mnb_city.best_score_)

###############################################################################################
# Write predictions to the output file
###############################################################################################
y_validation = np.column_stack((y_city, y_country))
io.write_prediction(io.VALIDATION_OUT, y_validation)

#y_test = np.column_stack((y_city, y_country))
#io.write_prediction(io.TESTING_OUT, y_test)
