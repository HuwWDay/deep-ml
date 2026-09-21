import re

def infer_matrix_shape(expr: str, shapes: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
    expr = expr.replace(" ", "")

    # --- Transpose ---
    if expr.endswith(".T"):
        var = expr[:-2]
        sh = shapes[var]
        return (sh[1], sh[0]) if len(sh) == 2 else (1,) if len(sh) == 0 else (sh[0],)

    # --- Matrix multiplication ---
    if "@" in expr:
        left, right = expr.split("@")
        A = shapes[left]
        B = shapes[right]

        # Handle vector @ matrix: (n,) @ (n,m) → (m,)
        if len(A) == 1 and len(B) == 2:
            return (B[1],)
        # Handle matrix @ vector: (n,m) @ (m,) → (n,)
        if len(A) == 2 and len(B) == 1:
            return (A[0],)
        # Standard matrix @ matrix
        return (A[0], B[1])

    # --- Elementwise ops ---
    for op in ["+", "-", "*", "/"]:
        if op in expr:
            left, right = expr.split(op)
            # scalars ignored; return array shape
            re