import numpy as np


def embedding_via_one_hot(token_ids, W):
  """Compute token embeddings via one-hot encoding and matrix multiplication.

  Args:
      token_ids: list or 1D array of integer token IDs
      W: numpy array of shape (vocab_size, embed_dim)

  Returns:
      numpy array of shape (len(token_ids), embed_dim)
  """
  vocab_size, embed_dim = W.shape
  seq_len = len(token_ids)

  # Create one-hot matrix: shape (seq_len, vocab_size)
  one_hot = np.zeros((seq_len, vocab_size), dtype=W.dtype)
  one_hot[np.arange(seq_len), token_ids] = 1.0

  # Matrix multiplication: (seq_len, vocab_size) @ (vocab_size, embed_dim) -> (seq_len, embed_dim)
  return one_hot @ W