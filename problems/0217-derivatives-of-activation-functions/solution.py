def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
    import numpy as np
	sig = 1 / (1+np.exp(-x))
    sigdiff = sig*(1-sig)

    tanh = (np.exp(x)- np.exp(-x)) / (np.exp(x) + np.exp(-x))
    tanhdiff = 1 - tanh**2 

    if x > 0:
        resdiff = 1
    else:
        resdiff = 0
    dict = {"sigmoid": sigdiff, "tanh": tanhdiff, "relu": resdiff}
    return dict