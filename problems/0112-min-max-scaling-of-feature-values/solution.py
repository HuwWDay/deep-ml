def min_max(x: list[int]) -> list[float]:
	# Your code here
	smol = min(x)
    big = max(x)
    if big == smol:
        return [0 for i in x]
    else:
        return [(i-smol)/(big-smol) for i in x]