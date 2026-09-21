import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	# Your code here
	out = w1 @ x
	out = np.maximum(out, 0)
	out = w2 @ out
	out = out + x
	out = np.maximum(out, 0)
	return out