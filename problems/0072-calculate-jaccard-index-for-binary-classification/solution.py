
import numpy as np

def jaccard_index(y_true, y_pred):
	n = len(y_true)
    inter, union = 0, 0
    for i in range(n):
        if y_true[i] == 1 and y_pred[i] == 1:
            inter += 1
        if y_true[i] == 1 or y_pred[i] == 1:
            union += 1
    if union == 0:
        result = 0
    result = inter/union
    

	return round(result, 3)
