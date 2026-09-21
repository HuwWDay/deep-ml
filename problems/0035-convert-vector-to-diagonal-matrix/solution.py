import numpy as np

def make_diagonal(x):
	array_len = np.shape(x)[0]
    diag = np.zeros((array_len, array_len))
    # for each row of diag
    for i in range(diag.shape[0]):
        diag[i][i] = x[i]
    return diag