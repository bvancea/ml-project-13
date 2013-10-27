__author__ = 'bogdan'

from sklearn import linear_model
from sklearn.metrics import metrics
from sklearn import cross_validation
import pandas as pd
import io_utils as io
import pylab as pl
from sklearn.svm import SVR
from sklearn import feature_selection
from sklearn.feature_selection import chi2

relevant_features = ['width', 'rob', 'iq', 'lsq', 'l2ucache', 'depth']
relevant_features = ['width', 'rob', 'iq', 'lsq', 'rfsize', 'rfread', 'rfwrite', 'l2ucache', 'depth']
# I get singular matrix with dis one
# relevant_features = ['width','rob','iq','lsq','rfsize','rfread','rfwrite','gshare','btb','branches','l1icache','l1dcache','l2ucache','depth']

#read input data
#training_X, training_y, testing_X, validation_X = io.read_regression_data()

training = io.read_x_y(io.TRAINING_IN)

training_y = training.delay

testing_X = io.read_x(io.TESTING_IN)
validation_X = io.read_x(io.VALIDATION_IN)

#scale the data
#scaler = StandardScaler()
#scaler.fit(training_X)
#training_X = scaler.transform(training_X)
#testing_X = scaler.transform(testing_X)
#validation_X = scaler.transform(validation_X)

## pd.scatter_matrix(training,alpha=0.5,figsize=(10,10), diagonal='hist')

#preprocess training
training_X = io.preprocess_features(training, relevant_features)
validation_X = io.preprocess_features(validation_X, relevant_features)

#cross validation
print "Cross validation"
for k in range(2, 8):
    print "k = " + `k`
    kf = cross_validation.KFold(len(training_X), k)
    for train_index, test_index in kf:
        X_train, X_test = training_X[train_index], training_X[test_index]
        y_train, y_test = training_y[train_index], training_y[test_index]
        regressor = linear_model.Ridge()
        regressor.fit(X_train, y_train)
        X_test_pred = regressor.predict(X_test)
        loss = metrics.mean_squared_error(y_test, X_test_pred)
        #print "loss = " + `loss`
        training_y_pred = regressor.predict(training_X)
        loss = metrics.mean_squared_error(training_y, training_y_pred)
        print loss

print "\nLoss for LR without cross validation"
regressor = linear_model.Ridge()
regressor.fit(training_X, training_y)
# see the loss function loss(true,pred)
training_y_pred = regressor.predict(training_X)
loss = metrics.mean_squared_error(training_y, training_y_pred)
print loss

#regularizers = [0.0001, 0.001, 0.05, 0.001, 0.05, 0.01, 0.1, 1.0, 5, 10, 100]

# feature selection
### with SelectKbest -- takes a lot
#featureSelector = feature_selection.SelectKBest(chi2, 3)
#Xtrunc = featureSelector.fit_transform(training_X, training_y)
### recursive -- takes a lot
#estimator = SVR(kernel="linear")
#selector = feature_selection.RFE(estimator, 3)
#selector = selector.fit(training_X, training_y)


#eliminate features one by one
print "\nFeatures in descending order"
features = relevant_features
while len(features) > 0:
    max_k = 0
    max_loss = 0
    for k in range(0, len(features) - 1):
        actual_features = features[:]
        del actual_features[k]
        actual_training_X = io.preprocess_features(training, actual_features)
        regressor.fit(actual_training_X, training_y)
        training_y_pred = regressor.predict(actual_training_X)
        loss = metrics.mean_squared_error(training_y, training_y_pred)
        # looking for the maximum loss so that to eliminate the corresponding feature
        if loss > max_loss:
            max_k = k
            max_loss = loss
    print features[max_k]
    del features[max_k]

#make predictions
regressor.fit(training_X, training_y)
training_y_pred = regressor.predict(training_X)
validation_y = regressor.predict(validation_X)

#write to output file
#io.write_prediction(TESTING_OUT, testing_y)
io.write_prediction(io.VALIDATION_OUT, validation_y)

