import numpy as np

def accuracy_score(y_true, y_pred):
	total = len(y_true)
    count = 0
    for i in range(total):
        if y_true[i] == y_pred[i]:
            count += 1
    return count/total