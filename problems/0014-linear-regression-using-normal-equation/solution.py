import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
    X = np.array(X)
    y = np.array(y)

    theta = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y

    theta = theta.tolist()
    theta = [round(x, 4) for x in theta]
	return theta