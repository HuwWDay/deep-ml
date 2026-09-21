def determinant_4x4(matrix: list[list[float]]) -> float:
    n = len(matrix)
    
    # Base cases
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(n):
        # Build the (n-1)x(n-1) minor matrix after removing row 0 and column col
        minor = [
            [matrix[r][c] for c in range(n) if c != col]
            for r in range(1, n)
        ]
        sign = (-1) ** col
        det += sign * matrix[0][col] * determinant_4x4(minor)

    return det
