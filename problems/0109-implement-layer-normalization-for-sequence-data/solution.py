import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	# mean & variance along last dimension
    mu = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)

    # normalize
    X_norm = (X - mu) / np.sqrt(var + epsilon)

    # scale + shift
    return gamma * X_norm + beta