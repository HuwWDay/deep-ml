import numpy as np

import numpy as np 

def svd_2x2(A: np.ndarray) -> tuple:
    B = A.T @ A  # symmetric 2×2

    # Use arctan2 for numerical stability
    if B[0, 0] == B[1, 1]:
        theta = np.pi / 4
    else:
        theta = 0.5 * np.arctan2(2 * B[0, 1], B[0, 0] - B[1, 1])

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    # Diagonalize
    D = R.T @ B @ R
    s = np.sqrt(np.maximum([D[0, 0], D[1, 1]], 0))   # ensure non-negative

    # Build Σ and its inverse (avoid division by zero)
    sig_inv = np.diag([1/s[0] if s[0] != 0 else 0,
                       1/s[1] if s[1] != 0 else 0])

    # Compute U
    U = A @ R @ sig_inv

    # Sort values descending
    idx = np.argsort(s)[::-1]
    s = s[idx]
    U = U[:, idx]
    Vt = R.T[idx, :]

    return U, s, Vt
