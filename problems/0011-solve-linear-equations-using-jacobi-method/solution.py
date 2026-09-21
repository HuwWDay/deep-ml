import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    x = np.zeros_like(b, dtype=float)
    
    for _ in range(n):
        x_new = np.zeros_like(x)
        for i in range(A.shape[0]):
            sigma = np.dot(A[i, :], x) - A[i, i] * x[i]
            x_new[i] = (b[i] - sigma) / A[i, i]
        x = x_new
    return x