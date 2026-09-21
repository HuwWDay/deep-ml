def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
    import numpy as np
    det = matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
    trace = matrix[0][0]+matrix[1][1]

    discrim = trace**2 - 4*det
    discrim = np.sqrt(discrim)

    sol1 = (trace+discrim)/2
    sol2 = (trace-discrim)/2
    if sol1 > sol2:
        return [sol1, sol2]
    else:
        return [sol2, sol1]
    