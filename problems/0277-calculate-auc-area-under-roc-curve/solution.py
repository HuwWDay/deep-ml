import numpy as np


def calculate_auc(y_true, y_scores):
    """Calculate the Area Under the ROC Curve (AUC).

    Args:
        y_true: List or array of binary ground truth labels (0 or 1)
        y_scores: List or array of predicted probabilities or confidence scores

    Returns:
        AUC value as a float
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    pos_mask = y_true == 1
    neg_mask = y_true == 0

    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)

    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "AUC is undefined when all samples belong to a single class."
        )

    pos_scores = y_scores[pos_mask]
    neg_scores = y_scores[neg_mask]

    score_sum = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                score_sum += 1.0
            elif p == n:
                score_sum += 0.5

    return score_sum / (n_pos * n_neg)