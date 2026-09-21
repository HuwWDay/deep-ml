import numpy as np

def batch_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	# Compute mean and variance per channel across batch, height, and width
    mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
    var = np.var(X, axis=(0, 2, 3), keepdims=True)

    # Normalize
    X_norm = (X - mean) / np.sqrt(var + epsilon)

    # Scale and shift (broadcast gamma and beta to match shape)
    out = gamma * X_norm + beta
    return out