import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
    # Check T inverible
    if np.linalg.det(T) == 0:
        return -1
    if np.linalg.det(S) == 0:
        return -1
    
    A = np.array(A)
    T = np.array(T)
    S = np.array(S)

    Tinv = np.linalg.inv(T)

    transformed_matrix = Tinv @ A @ S
	return transformed_matrix