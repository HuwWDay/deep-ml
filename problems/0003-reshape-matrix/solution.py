import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
    a = np.array(a)
    a_total = np.shape(a)[0]*np.shape(a)[1]
    new_shape_total = new_shape[0]*new_shape[1]
    
    if a_total != new_shape_total:
        return []
    else:

   

        reshaped_matrix = np.reshape(a, new_shape)
        reshaped_matrix = reshaped_matrix.tolist()

        return reshaped_matrix