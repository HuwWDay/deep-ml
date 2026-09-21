import numpy as np

def simulate_clt(num_samples: int, sample_size: int, distribution: str = 'uniform') -> float:
	"""
	Simulate the Central Limit Theorem (CLT).

	Args:
		num_samples: number of repeated samples to draw
		sample_size: size of each sample
		distribution: 'uniform' or 'exponential'

	Returns:
		Mean of the sample means (float)
	"""
    means = []
    for _ in range(num_samples):
	    if distribution == "uniform":
            samples = np.random.uniform(size = sample_size)
        elif distribution == "exponential":
            samples = np.random.exponential(size = sample_size)
        means.append(np.mean(samples))
    
    return np.mean(means)