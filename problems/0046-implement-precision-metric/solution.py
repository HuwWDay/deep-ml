import numpy as np
def precision(y_true, y_pred):
	# Your code here
	# TP: (1, 1)
    # FP: (0, 1)
    TP, FP = 0, 0
    for i in range(len(y_true)):
        if y_pred[i] == 1:
            if y_true[i] == 1:
                TP += 1
            else:
                FP += 1
    return TP / (TP + FP)
