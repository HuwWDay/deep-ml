import numpy as np

def overlapping_max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """
    Applies overlapping max pooling to a 4D tensor (N, C, H, W).

    Args:
        x: Input array of shape (N, C, H, W)
        kernel_size: Size of pooling window
        stride: Stride between pooling windows

    Returns:
        A 4D tensor after overlapping max pooling.
    """
    N, C, H, W = x.shape
    k = kernel_size
    s = stride

    # Compute output spatial dimensions
    out_h = (H - k) // s + 1
    out_w = (W - k) // s + 1

    # Allocate output
    y = np.zeros((N, C, out_h, out_w))

    # Perform overlapping max pooling
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * s
            w_start = j * s
            window = x[:, :, h_start:h_start+k, w_start:w_start+k]
            y[:, :, i, j] = window.max(axis=(2, 3))

    return y
