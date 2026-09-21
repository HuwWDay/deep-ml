import numpy as np

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            if y_true[i] == 1:
                TP += 1
            else: 
                TN += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        else:
            FN += 1
    if TP + FN == 0:
        recall = 0.0
    else:
        recall = TP / (TP + FN)
    precision = TP / (TP + FP)

    if precision == recall == 0:
        return 0.0
    else:

        F = (1+beta**2)*precision*recall / ((beta**2 * precision) + recall)
        return round(F, 3)
