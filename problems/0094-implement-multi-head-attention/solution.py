import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """Compute query, key, and value matrices."""
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V


import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def self_attention(Q, K, V):
    """Compute scaled dot-product attention for a single head."""
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    attn_weights = softmax(scores)
    return attn_weights @ V

def multi_head_attention(Q, K, V, n_heads):
    """Multi-head attention given precomputed Q, K, V."""
    d_model = Q.shape[-1]
    assert d_model % n_heads == 0, "Feature dimension must divide evenly into n_heads"
    d_head = d_model // n_heads

    # Split Q, K, V into heads
    Q_heads = np.split(Q, n_heads, axis=-1)
    K_heads = np.split(K, n_heads, axis=-1)
    V_heads = np.split(V, n_heads, axis=-1)

    # Com