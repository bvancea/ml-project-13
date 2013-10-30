import numpy as np
import pandas as pd
from sklearn import preprocessing, linear_model, feature_selection
from itertools import compress


def do_feature_selection(reg, training, validation_X, testing_X):

    selector = feature_selection.RFECV(estimator=reg, cv=2, verbose=0)
    training_y = training.delay
    training_X = training.drop(labels=['delay'], axis=1)

    selector.fit(training_X, training_y)
    a = list(training_X)
    b = list(compress(training_X, selector.support_))
    c = [x for x in a if x not in b]

    training = training.drop(labels=c, axis=1)
    validation_X = validation_X.drop(labels=c, axis=1)
    testing_X = testing_X.drop(labels=c, axis=1)

    return training, validation_X, testing_X


def preprocess_data(training, validation_X, testing_X, normalize=False, standardize=False):

    if normalize:
        #perform standardization on the features
        scaler = DataFrameStandardizer()
        training_y = training.delay
        training_X = training.drop(labels=['delay'], axis=1)
        training_X = scaler.fit_transform(training_X)
        validation_X = scaler.transform(validation_X)
        testing_X = scaler.transform(testing_X)

        #rebuild feature matrix
        training = training_X.copy()
        training['delay'] = training_y

    if normalize:
        #perform standardization on the features
        scaler = DataFrameStandardizer(scaler=preprocessing.StandardScaler())
        training_y = training.delay
        training_X = training.drop(labels=['delay'], axis=1)
        training_X = scaler.fit_transform(training_X)
        validation_X = scaler.transform(validation_X)
        testing_X = scaler.transform(testing_X)

        #rebuild feature matrix
        training = training_X.copy()
        training['delay'] = training_y

    return training, validation_X, testing_X


class DataFrameStandardizer:
    """
        Simple wrapper over the scikit-learn StandardScaler so that it works
        properly with pandas DataFrame objects
    """
    def __init__(self, scaler=preprocessing.MinMaxScaler()):
        self.scaler = scaler

    def fit_transform(self, data):
        scaled_data = self.scaler.fit_transform(data)
        return pd.DataFrame(scaled_data, index=data.index, columns=data.columns)

    def transform(self, data):
        scaled_data = self.scaler.transform(data)
        return pd.DataFrame(scaled_data, index=data.index, columns=data.columns)

    def inverse_transform(self, data):
        scaled_data = self.scaler.inverse_transform(data, copy=True)
        return pd.DataFrame(scaled_data, index=data.index, columns=data.columns)

class PolyModel:

    def transform(self, X_train, order):
        labels = [x for x in X_train]
        features_x = np.vander(X_train[labels[0]], order)
        for label in labels[1:]:
            current_x = np.vander(X_train[label], order)
            features_x = np.concatenate((current_x, features_x), axis=1)

        return features_x


class PolyRegressor:

    def __init__(self, degree, regressor):
        self.degree = degree
        self.regressor = regressor
        self.poly_model = PolyModel()

    def fit(self, X_train, y_train):
        X_train = self.poly_model.transform(X_train, self.degree)
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        X_test = self.poly_model.transform(X_test, self.degree)
        result = self.regressor.predict(X_test)
        return result


class FeatureAdder:

    def __init__(self, functions, corr_bound=0.1, if_corr_bound=0.3, replace=False, intra_corr_func=np.sum):
        self.corr_bound = corr_bound
        self.if_corr_bound = if_corr_bound
        self.functions = functions
        self.replace = replace
        self.columns = []
        self.inter_corr_features = []
        self.intra_corr_func = intra_corr_func

    def kernelize(self, features_df, feature_y_label, intra_feature_corr=True):
        """
            Adds new features using the correlation matrix of the existing features.
        """
        correlation_matrix = features_df.corr()
        x_labels = filter(lambda x: x != 'delay', features_df)

        #copy the data frame
        new_features_df = features_df.copy()

        for function in self.functions:
            for label in x_labels:
                abs_corr = abs(correlation_matrix[feature_y_label][label])

                if abs_corr > self.corr_bound:
                    self.columns.append(label)
                    #funny python ternary operator
                    new_label = label if self.replace else function.__name__ + "_" + label
                    new_features_df[new_label] = function(features_df[label])

        x_labels = filter(lambda x: x != 'delay', features_df)
        if intra_feature_corr:
            for i in x_labels:
                for j in x_labels:
                    if i != j:
                        abs_corr = abs(correlation_matrix[i][j])
                        if abs_corr > self.corr_bound * 3:
                            f = self.intra_corr_func
                            new_tuple = tuple([f, i, j])
                            self.inter_corr_features.append(new_tuple)
                            #funny python ternary operator
                            new_label = i + f.__name__ + j
                            new_features_df[new_label] = features_df[i] + features_df[j]

        return new_features_df

    def transform(self, features_df, intra_feature_corr=True):
        """
            Apply the input transformation to another set
        """
        new_features_df = features_df.copy()

        #added features from y correlation
        for function in self.functions:
            for label in self.columns:
                new_label = label if self.replace else function.__name__ + "_" + label
                new_features_df[new_label] = function(features_df[label])

        if intra_feature_corr:
            #added features from inter-feature correlation
            for tup in self.inter_corr_features:
                f = tup[0]
                i = tup[1]
                j = tup[2]
                new_label = i + f.__name__ + j
                new_features_df[new_label] = f(features_df[i], features_df[j])

        return new_features_df


def sum_squares(x,y):
    return x**2 + y**2


def product(x,y):
    return x*y


def cube(x):
    return np.power(x, 3)


def quad(x):
    return np.power(x, 4)