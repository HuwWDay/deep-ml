import numpy as np

def rref(matrix):
    """
    Converts a matrix into its Reduced Row Echelon Form (RREF) using
    Gauss-Jordan Elimination.
    """
    # 1. Initialization and Data Preparation
    A = matrix.astype(float).copy() # Work with a copy and float data type
    rows, cols = A.shape
    pivot_row = 0
    
    # 2. Forward Elimination (Gaussian Elimination to REF)
    # This loop works from the top-left to the bottom-right
    for col in range(cols):
        if pivot_row >= rows:
            break
        
        # Find the best pivot (largest absolute value in the current column below pivot_row)
        pivot_candidate = np.argmax(np.abs(A[pivot_row:, col])) + pivot_row
        
        # If the column is effectively zero, skip to the next column
        if np.isclose(A[pivot_candidate, col], 0):
            continue
            
        # Swap the current pivot_row with the row containing the largest element
        A[[pivot_row, pivot_candidat