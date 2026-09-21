import numpy as np

# We reuse the Gaussian Elimination function to find pivot columns
def row_echelon_form(matrix):
    """
    Helper function: Converts a matrix to Row Echelon Form (REF).
    Returns the REF matrix and a list of the column indices containing pivots.
    """
    A = matrix.astype(float).copy()
    rows, cols = A.shape
    pivot_row = 0
    pivot_cols = []
    
    for col in range(cols):
        if pivot_row >= rows:
            break
        
        # 1. Partial Pivoting: Find the row with the largest absolute value
        pivot_candidate = np.argmax(np.abs(A[pivot_row:, col])) + pivot_row
        
        # Check if the column is effectively zero
        if np.isclose(A[pivot_candidate, col], 0):
            continue
            
        # 2. Swap rows
        A[[pivot_row, pivot_candidate]] = A[[pivot_candidate, pivot_row]]
        
        # Store the pivot column index
        pivot_cols.append(col)
        
        # 3. Eliminati