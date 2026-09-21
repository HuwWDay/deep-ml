import math

def softmax(scores: list[float]) -> list[float]:
    import math
	probabilities = [math.exp(z) for z in scores]
    prob_sum = sum(probabilities)
    probabilities = [z/prob_sum for z in probabilities]
	return probabilities