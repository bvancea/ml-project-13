__author__ = 'bogdan'

from sklearn import linear_model
import pandas as pd
import io_utils as io


relevant_features = ['width', 'rob', 'iq', 'lsq', 'l2ucache', 'depth']

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
pd.scatter_matrix(training,alpha=0.5,figsize=(10,10), diagonal='hist')

#preprocess training
training_X = io.preprocess_features(training, relevant_features)
validation_X = io.preprocess_features(validation_X, relevant_features)

#train regressor
#kf = KFold(len(training_X), n_folds=10)
regularizers = [0.0001, 0.001, 0.05, 0.001, 0.05, 0.01, 0.1, 1.0, 5, 10, 100]

regressor = linear_model.Ridge()
regressor.fit(training_X, training_y)


#make predictions
validation_y = regressor.predict(validation_X)

#write to output file
#io.write_prediction(TESTING_OUT, testing_y)
io.write_prediction(io.VALIDATION_OUT, validation_y)

