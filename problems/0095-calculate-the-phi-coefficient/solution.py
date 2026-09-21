def phi_corr(x: list[int], y: list[int]) -> float:
	"""
	Calculate the Phi coefficient between two binary variables.

	Args:
	x (list[int]): A list of binary values (0 or 1).
	y (list[int]): A list of binary values (0 or 1).

	Returns:
	float: The Phi coefficient rounded to 4 decimal places.
	"""
	# Your code here
    import numpy as np
	x_00, x_01, x_10, x_11 = 0, 0, 0, 0
    n = len(x)
    for i in range(n):
        if x[i] == y[i]:
            if x[i] == 0:
                x_00 += 1
            else:
                x_11 +=1
        else:
            if x[i] == 0:
                x_01 += 1
            else:
                x_10 +=1
    num = (x_00 * x_11) - (x_01 * x_10)
    denom = (x_00 + x_01) * (x_10 + x_11) * (x_00 + x_10) * (x_01 + x_11)
    val = num/np.sqrt(denom)
	return round(val,4)