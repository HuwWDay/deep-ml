def swish(x: float) -> float:
	"""
	Implements the Swish activation function.

	Args:
		x: Input value

	Returns:
		The Swish activation value
	"""
	import numpy as np
    val = x / (1+np.exp(-x))
    return val