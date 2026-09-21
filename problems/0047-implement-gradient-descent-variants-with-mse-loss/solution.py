import numpy as np

def gradient_descent(X, y, weights, learning_rate=0.01, n_iterations=1000, batch_size=1, method='batch'):
    n_samples = len(y)

    for _ in range(n_iterations):
        if method == "batch":
            # Compute predictions
            pred = np.dot(X, weights)
            # Compute gradient
            gradient = (2 / n_samples) * np.dot(X.T, (pred - y))
            # Update weights
            weights -= learning_rate * gradient

        elif method == "stochastic":
            for i in range(n_samples):
                xi = X[i:i+1]
                yi = y[i]
                pred_i = np.dot(xi, weights)
                gradient = 2 * np.dot(xi.T, (pred_i - yi))
                weights -= learning_rate * gradient

        elif method == "mini_batch":
            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                pred_batch = np.dot(X_batch, w