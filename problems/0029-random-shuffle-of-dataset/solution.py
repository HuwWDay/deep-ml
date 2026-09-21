import numpy as np

def shuffle_data(X, y, seed=None):
    np.random.seed(seed)
    index = np.random.permutation(len(y))
    X_new = X[index]
    y_new = y[index]
    return X_new, y_new

