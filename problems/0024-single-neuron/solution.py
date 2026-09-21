import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Forward pass 
    z = [sum(i[0] * i[1] for i in zip(features[k], weights)) for k in range(len(features))]
    z = [element + bias for element in z]
    probabilities = [1/(1+math.exp(-k)) for k in z]
    
    # MSE
    n = len(labels)
    Err = [(pair[0]-pair[1])**2 for pair in zip(probabilities, labels)]
    mse = 1/float(n)*sum(Err)
	return probabilities, mse