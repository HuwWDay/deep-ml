import numpy as np

def fn(seed, mean, std, n, bins):
    # return (counts, edges) as plain Python lists
    np.random.seed(seed)
    normal = np.random.normal(mean, std, n)
    return np.histogram(normal, bins=bins)
