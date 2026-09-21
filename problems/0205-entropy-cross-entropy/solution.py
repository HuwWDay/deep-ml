import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""
	# Your code here
	ent = - sum([p*np.log(p+10**-10) for p in P])
    ent = abs(ent)
    crossent = -sum(p * np.log(q+10**-10) for p, q in zip(P, Q))
    return ent, crossent