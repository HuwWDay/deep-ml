import numpy as np

def mutual_information(joint_prob: list[list[float]]) -> float:
    """
    Compute the mutual information between two random variables.
    
    Args:
        joint_prob: 2D joint probability distribution P(X,Y)
    
    Returns:
        Mutual information I(X;Y)
    """
    joint_prob = np.array(joint_prob, dtype=float)

    # Marginals
    px = np.sum(joint_prob, axis=1)  # sum rows
    py = np.sum(joint_prob, axis=0)  # sum columns

    I = 0.0
    for x in range(len(px)):
        for y in range(len(py)):
            pxy = joint_prob[x, y]
            if pxy > 0:  # avoid log(0)
                I += pxy * np.log(pxy / (px[x] * py[y]))

    return I
