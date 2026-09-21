import numpy as np

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray,
                 initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    
    mse_values = []
    n = len(labels)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    weights = initial_weights.copy()
    bias = initial_bias

    for _ in range(epochs):
        # Forward pass
        z = np.dot(features, weights) + bias
        guess = sigmoid(z)

        # Compute MSE
        mse = np.mean((guess - labels) ** 2)
        mse_values.append(mse)

        # Compute gradients (vectorized)
        d_mse_dz = (2 / n) * (guess - labels) * guess * (1 - guess)
        grad_w = np.dot(features.T, d_mse_dz)
        grad_b = np.sum(d_mse_dz)

        # Update weights and bias
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return weights, bias, mse_values
