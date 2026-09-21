
import numpy as np

def r_squared(y_true, y_pred):
	# Write your code here
	SSR, SST = 0, 0
    n = len(y_true)
    mean = sum(y_true)/n
    for i in range(n):
        SSR += (y_true[i]-y_pred[i])**2
        SST += (y_true[i] - mean)**2
    return 1-SSR/SST


