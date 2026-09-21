import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Performs linear regression using gradient descent.
    
    Parameters:
        X (np.ndarray): Feature matrix of shape (m, n)
        y (np.ndarray): Target vector of shape (m,) or (m, 1)
        alpha (float): Learning rate
        iterations (int): Number of iterations
    
    Returns:
        np.ndarray: Final parameter vector theta (rounded to 4 decimals)
    """
    m, n = X.shape
    y = y.reshape(-1, 1)  # Ensure column vector
    theta = np.zeros((n, 1))  # Initialize parameters
    
    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1 / m) * X.T.dot(errors)
        theta -= alpha * gradient

    # Round final theta values
    return np.round(theta, 4)
