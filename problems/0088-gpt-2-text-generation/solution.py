import numpy as np

# --- Helper Functions for Model Components ---

def layer_norm(x, g, b, eps=1e-5):
    """
    Applies layer normalization to the input tensor x.
    'g' is the gain (scale) and 'b' is the bias (shift).
    """
    # Calculate mean and variance along the last dimension (features)
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    
    # Normalize
    x_norm = (x - mean) / np.sqrt(variance + eps)
    
    # Scale and shift
    return x_norm * g + b

def gpt2_simplified_forward_pass(input_tokens: list, hparams: dict, params: dict):
    """
    Runs a forward pass of the simplified GPT-2 model based on the 'params'.
    
    This function incorporates:
    1. Token Embeddings (params["wte"])
    2. Positional Embeddings (params["wpe"])
    5. Layer Normalization (params["ln_f"])
    
    It also shows where components 3 (Multi-head Attention) and 4 (Feed-Forward)
    would be applied if 'para