import numpy as np

def local_response_normalization(
    x: np.ndarray,
    n: int = 5,
    k: float = 2.0,
    alpha: float = 1e-4,
    beta: float = 0.75
) -> np.ndarray:
    """
    Applies Local Response Normalization (LRN) across the channel dimension.

    Args:
        x: Input tensor of shape (N, C, H, W)
        n: Local window size (number of channels to normalize over)
        k: Additive constant (usually 2)
        alpha: Scaling parameter
        beta: Exponent parameter

    Returns:
        Normalized tensor of the same shape as input.
    """
    N, C, H, W = x.shape
    half_n = n // 2

    # Square of input
    squared = x ** 2

    # Accumulator for the denominator
    scale = np.zeros_like(x)

    # Compute normalization denominator
    for c in range(C):
        # Determine channel window
        c_start = max(0, c - half_n)
        c_end = min(C, c + half_n + 1)
        # Sum across the channel window
        scale[:, c, :, :