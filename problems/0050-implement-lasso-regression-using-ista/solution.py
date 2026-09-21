import numpy as np

def l1_regularization_gradient_descent(X: np.array, y: np.array, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4) -> tuple:
	n_samples, n_features = X.shape

	weights = np.zeros(n_features)
	bias = 0
	
    for _ in range(max_iter):
        # Predictions
        y_pred = X @ weights + bias

        # Compute residuals
        residuals = y_pred - y

        # Compute gradients
        bias_grad = np.mean(residuals)
        weight_grad = (X.T @ residuals) / n_samples + alpha * np.sign(weights)

        # Update parameters
        new_weights = weights - learning_rate * weight_grad
        new_bias = bias - learning_rate * bias_grad

        # Check convergence
        if np.linalg.norm(weight_grad, ord=1) < tol:
            weights, bias = new_weights, new_bias
            break

        weights, bias = new_weights, new_bias

    return weights, bias