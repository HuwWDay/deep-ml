import numpy as np

def phi_transform(data: list[float], degree: int) -> list[list[float]]:
	"""
	Perform a Phi Transformation to map input features into a higher-dimensional space by generating polynomial features.

	Args:
		data (list[float]): A list of numerical values to transform.
		degree (int): The degree of the polynomial expansion.

	"""
    out = [[0 for _ in range(degree+1)] for _ in range(len(data))]

    for d in range(len(data)):
        for i in range(degree+1):
            out[d][i] = data[d] ** i

    return out

