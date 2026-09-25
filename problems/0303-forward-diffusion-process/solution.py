import numpy as np

def forward_diffusion(
    x_0: np.ndarray, 
    t: int, 
    beta_start: float, 
    beta_end: float, 
    num_timesteps: int, 
    noise: np.ndarray
) -> np.ndarray:
    """
    Apply forward diffusion process to add noise to input data.
    
    Args:
        x_0: Original input data (numpy array)
        t: Timestep (1-indexed, from 1 to num_timesteps)
        beta_start: Starting value of linear beta schedule
        beta_end: Ending value of linear beta schedule
        num_timesteps: Total number of diffusion timesteps
        noise: Noise array (same shape as x_0)
    
    Returns:
        Noisy sample x_t as numpy array
    """
    # 1. Linear beta schedule from beta_start to beta_end over num_timesteps
    betas = np.linspace(beta_start, beta_end, num_timesteps)
    
    # 2. alpha_t = 1 - beta_t
    alphas = 1.0 - betas
    
    # 3. Cumulative product: alpha_bar_t = prod_{s=1}^t alpha_s
    alphas_cumprod = np.cumprod(alphas)
    
    # 4. Extract alpha_bar for timestep t (t is 1-indexed -> index t - 1)
    alpha_bar_t = alphas_cumprod[t - 1]
    
    # 5. Closed-form sample x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    x_t = np.sqrt(alpha_bar_t) * x_0 + np.sqrt(1.0 - alpha_bar_t) * noise
    
    return x_t