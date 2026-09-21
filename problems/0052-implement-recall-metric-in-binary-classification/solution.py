import numpy as np
def recall(y_true, y_pred):
    # TP: 1, 1
    # FN: 1, 0
    TP, FN = 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                TP += 1
            else:
                FN += 1
    total = TP + FN
    if total == 0:
        return 0.0
    else:
        return round(TP / total, 3)
