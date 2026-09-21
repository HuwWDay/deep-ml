def compressed_col_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

	:param dense_matrix: List of lists representing the dense matrix
	:return: Tuple of (values, row indices, column pointer)
	"""
    import numpy as np
	dense_matrix = np.array(dense_matrix)
    values, rows, columns = [], [], [0]
    for column in range(dense_matrix.shape[1]):
        for row in range(dense_matrix.shape[0]):

            if dense_matrix[row][column] != 0:
                rows.append(row)
                values.append(dense_matrix[row][column])
        columns.append(len(values))
    return values, rows, columns
