
def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
	n = len(actual)
    TP, FN, FP, TN = 0, 0, 0, 0
    for i in range(n):
        if actual[i] == predicted[i]:
            if actual[i] == 1:
                TP += 1
            else:
                TN += 1
        elif actual[i] == 0:
            FP += 1
        else:
            FN += 1
    
    confusion_matrix = [[TP, FN], [FP, TN]]

    accuracy = (TP+TN) / (TP+TN+FP+FN)

    precision = TP / (TP+FP)

    negativePredictive = TN / (TN + FN)

    recall = TP / (TP + FN)

    specificity = TN / (TN + FP)

    f1 = 2*(precision*recall) / (precision+recall)

    
	return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
