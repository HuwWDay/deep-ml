import numpy as np

def softmax(x):
    """Compute row-wise softmax with numerical stability."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def pattern_weaver(n, crystal_values, dimension):
    """Compute pattern weaving transformation."""
    length = len(crystal_values)
    crystal_values = np.array(crystal_values, dtype=float)

    # Compute interaction scores (like attention logits)
    score = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            score[i][j] = crystal_values[i] * crystal_values[j] / np.sqrt(dimension)
    
    # Apply softmax to each row
    score = softmax(score)
    
    # Weighted combination (like attention output)
    x = np.zeros(length)
    for i in range(length):
        for j in range(length):
            x[i] += score[i][j] * crystal_values[j]

    return np.round(x, 3)
