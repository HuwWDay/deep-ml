def transform_basis(B: list[list[float]], C: list[list[float]]) -> list[list[float]]:
    import numpy as np

    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)

    # Compute change-of-basis matrix
    P = np.linalg.inv(C) @ B

    return P.tolist()
