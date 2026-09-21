import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	# Your code here
    import math
	prob = [math.exp(s) for s in scores]
    sum_prob = sum(prob)
    logsoftmax = [math.log(s/sum_prob) for s in prob]
    return logsoftmax