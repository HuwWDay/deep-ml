def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if mode == "row":
        means = [sum(x)/len(x) for x in matrix]
    elif mode == "column":
        means = [sum(idx)/len(idx) for idx in zip(*matrix)]
        
	return means