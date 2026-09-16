import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    # Ensure float dtype so perturbation doesn't truncate on integer arrays
    x = x.astype(np.float64, copy=True)
    num_grad = np.zeros_like(x)
    
    # Iterate through every element of the input array
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig_val = x[idx]
        
        # Perturb coordinate in the positive direction
        x[idx] = orig_val + epsilon
        fx_plus = f(x)
        
        # Perturb coordinate in the negative direction
        x[idx] = orig_val - epsilon
        fx_minus = f(x)
        
        # Reset coordinate to original value
        x[idx] = orig_val
        
        # Two-sided finite difference
        num_grad[idx] = (fx_plus - fx_minus) / (2.0 * epsilon)
        
        it.iternext()

    # Compute relative error
    numerator = np.linalg.norm(num_grad - analytical_grad)
    denominator = np.linalg.norm(num_grad) + np.linalg.norm(analytical_grad)
    
    # Avoid zero division
    relative_error = numerator / np.maximum(1e-12, denominator)
    
    return num_grad, relative_error