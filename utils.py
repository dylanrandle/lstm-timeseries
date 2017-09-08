''' 
Data loading utilities
'''

import pandas as pd
import numpy as np

def read_timeseries(data_file):
    X = np.asarray(pd.read_csv(data_file, usecols=['Adj Close']))

    # split data
    num_test_samples = int(len(X)*0.1)
    X_train = X[:-num_test_samples]
    X_test = X[-num_test_samples:]

    # the output at time t should be the value of the next day, at t+1
    Y_train = X_train[1:]
    Y_test = X_test[1:]

    # get rid of last day of data, as we don't have a prediction
    X_train = X_train[:-1]
    X_test = X_test[:-1]

    assert X_train.shape == Y_train.shape

    return X_train, X_test, Y_train, Y_test