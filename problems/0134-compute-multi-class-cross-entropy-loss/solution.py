import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    total = 0
    for c in range(true_labels.shape[1]):
        for n in range(true_labels.shape[0]):
            total += true_labels[n][c]*np.log(predicted_probs[n][c]+epsilon)
    total /= -true_labels.shape[0]
    if int(10*total) == 0:
        return 0.0
    return total
