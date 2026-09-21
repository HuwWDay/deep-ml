import math

def sigmoid(z: float) -> float:
	#Your code here
    import math
	return round(1/(1+math.exp(-z)), 4)