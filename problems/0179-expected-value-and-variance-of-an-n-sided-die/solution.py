def dice_statistics(n: int) -> tuple[float, float]:
	"""
	Compute the expected value and variance of a fair n-sided die roll.

	Args:
		n (int): Number of sides of the die

	Returns:
		tuple: (expected_value, variance)
	"""
	mean = sum([1/float(n) * (i+1) for i in range(n)])
    exsquare = sum([1/float(n)*(i+1)*(i+1) for i in range(n)])
    var = exsquare - mean**2
    return mean, var