import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
    """
    Performs global average pooling over the spatial dimensions (H, W).
    Args:
        x (np.ndarray): Input array of shape (H, W, C)
    Returns:
        np.ndarray: Pooled vector of shape (C,)
    """
    return np.mean(x, axis=(0, 1))
