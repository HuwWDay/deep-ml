import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

	:param dense_matrix: 2D list representing a dense matrix
	:return: A tuple containing (values array, column indices array, row pointer array)
	"""
    dense_matrix = np.array(dense_matrix)
	values, columns, rows = [], [], [0]
    for row in range(dense_matrix.shape[0]):
        for column in range(dense_matrix.shape[1]):
            if dense_matrix[row][column] != 0:
                values.append(dense_matrix[row][column])
                columns.append(column)
        rows.append(len(values))
    return values, columns, rows
