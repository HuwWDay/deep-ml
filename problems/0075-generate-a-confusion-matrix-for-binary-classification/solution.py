
from collections import Counter

def confusion_matrix(data):
	# Implement the function here
	TP, FN, FP, TN = 0, 0, 0, 0
    for x in data:
        if x[0]==x[1]==1:
            TP += 1
        elif x[0]==x[1]==0:
            TN += 1
        elif x[0]==0 and x[1]==1:
            FP += 1
        else:
            FN += 1
    return [[TP, FN], [FP, TN]]
