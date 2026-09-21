import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
    if x_ini is None:
        x_ini = np.zeros_like(b)
    x_new = x_ini
    for _ in range(n):
        for i in range(len(b)):
            below = sum([A[i][j]*x_ini[j] for j in range(len(b)) if j > i])
            above = sum([A[i][j]*x_new[j] for j in range(len(b)) if j < i])
            x_new[i] = 1/A[i][i] * (b[i] - below - above)
        x_ini = x_new    

	return x_new
