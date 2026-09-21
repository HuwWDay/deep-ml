def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    result = [[scalar*matrix[j][i] for i in range(len(matrix))] for j in range(len(matrix[0]))]
	return result