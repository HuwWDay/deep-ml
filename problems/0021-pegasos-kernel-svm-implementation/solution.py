import numpy as np
def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100,sigma=1.0) -> (list, float):
	n = len(labels)
    b = 0
    alphas = np.zeros(n)
    if kernel == 'linear':
        def K(x, y):
            return np.dot(x, y)
    elif kernel == 'rbf':
        def K(x, y):
            return np.exp(-np.linalg.norm(x-y)**2 / (2*sigma**2))
    for t in range(iterations):
        lr = 1/(lambda_val*(t+1))
        for i in range(n):
            dec = 0
            for j in range(n):
                dec += alphas[j]*labels[j]*K(data[j], data[i])
            dec += b
            if labels[i]*dec < 1:
                alphas[i] += lr*(labels[i]-lambda_val*alphas[i])
                b += lr*labels[i]

	return alphas, b