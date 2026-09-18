import numpy as np

def superposition_reconstruct(W, b, X):
    """
    Compute reconstructed features for the toy superposition model.

    Args:
        W: array of shape (n_hidden, n_features)
        b: array of shape (n_features,)
        X: array of shape (batch_size, n_features)

    Returns:
        list of lists of shape (batch_size, n_features) with reconstructed features
    """
    W = np.asarray(W)
    X = np.asarray(X)
    b = np.asarray(b)
    
    # Corrected matrix multiplication order: (batch_size, n_features) @ (n_features, n_hidden) @ (n_hidden, n_features)
    reconstructed = np.maximum(X @ W.T @ W + b, 0)
    
    # Convert ndarray to list of lists to match docstring return type
    return reconstructed.tolist()