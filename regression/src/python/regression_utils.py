import numpy as np
import pandas as pd
from sklearn import preprocessing, feature_selection, cross_validation, metrics
from itertools import compress
from operator import itemgetter


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


class MultipleEstimatorCV:

    def __init__(self, k):
        self.k = k
        self.analyzed_models = []

    def cross_validate_models(self, models, X, y):

        test_size = 1.0/self.k
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
                    new_features_df[new_label] = features_df[label] * function(features_df[label])

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
                            new_features_df[new_label] = f(features_df[i], features_df[j])

        return new_features_df

    def transform(self, features_df, intra_feature_corr=True):
        """
            Apply the input transformation to another set
        """
        new_features_df = features_df.copy()
        x_labels = filter(lambda x: x != 'delay', features_df)
        #added features from y correlation
        for function in self.functions:
            for label in self.columns:
                new_label = label if self.replace else function.__name__ + "_" + label
                new_features_df[new_label] = features_df[label] * function(features_df[label])

        if intra_feature_corr:
            #added features from inter-feature correlation
            for tup in self.inter_corr_features:
                f = tup[0]
                i = tup[1]
                j = tup[2]
                new_label = i + f.__name__ + j
                new_features_df[new_label] = f(features_df[i], features_df[j])

        return new_features_df


def sum_squares(x, y):
    #scaler = preprocessing.MinMaxScaler()
    #x = pd.Series(data=scaler.fit_transform(x), index=x.index, name=x.name)
    #y = pd.Series(data=scaler.fit_transform(y), index=y.index, name=y.name)
    return np.exp(np.sin(x) + np.sin(y))


def product(x, y):
    return x*y


def cube(x):
    return np.power(x, 3)


def quad(x):
    return np.power(x, 4)