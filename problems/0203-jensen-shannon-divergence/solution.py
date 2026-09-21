import numpy as np

def jensen_shannon_divergence(P: list[float], Q: list[float]) -> float:
    """
    Compute the Jensen-Shannon Divergence between two probability distributions.
    
    Args:
        P: First probability distribution
        Q: Second probability distribution
    
    Returns:
        Jensen-Shannon Divergence value
    """
    P = np.array(P, dtype=float)
    Q = np.array(Q, dtype=float)

    # Normalize in case inputs are not perfectly normalized
    P /= np.sum(P)
    Q /= np.sum(Q)

    # Mixture distribution
    M = 0.5 * (P + Q)

    # Add small epsilon for numerical stability
    eps = 1e-12
    P_safe = P + eps
    Q_safe = Q + eps
    M_safe = M + eps

    DPM = np.sum(P_safe * np.log(P_safe / M_safe))
    DQM = np.sum(Q_safe * np.log(Q_safe / M_safe))

    return 0.5 * (DPM + DQM)
