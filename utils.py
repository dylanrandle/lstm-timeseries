''' 
Data loading utilities
'''

import pandas as pd
import numpy as np

def read_timeseries(data_file):
    timeseries = np.asarray(pd.read_csv(data_file, usecols=['Adj Close']))

    # Y is the next-day value of X
    X = timeseries[:-1]
    Y = timeseries[1:]

    assert X.shape == Y.shape

    return X, Y