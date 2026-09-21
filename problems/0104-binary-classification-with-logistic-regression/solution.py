import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """
    Implements binary classification prediction using Logistic Regression.

    Args:
        X: Input feature matrix (shape: N x D)
        weights: Model weights (shape: D)
        bias: Model bias

    Returns:
        Binary predictions (0 or 1)
    """
    # Linear combination
    z = X @ weights + bias
    z = np.clip(z, -500, 500)

    # Sigmoid activation
    sig = 1 / (1 + np.exp(-z))

    # Apply threshold (>= 0.5 → 1, else 0)
    preds = (sig >= 0.5).astype(int)

    return preds
