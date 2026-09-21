import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
	Q = np.dot(X, W_q)
	K = np.dot(X, W_k)
	V = np.dot(X, W_v)
	return Q, K, V

def self_attention(Q, K, V):
    # Compute raw attention scores
    scores = np.dot(Q, K.T)
    dk = K.shape[1]
    scores = scores / np.sqrt(dk)

    # Stable softmax (row-wise)
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Compute weighted sum of values
    output = np.dot(weights, V)
    return output
