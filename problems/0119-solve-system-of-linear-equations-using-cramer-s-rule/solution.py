import numpy as np

def cramers_rule(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    detA = np.linalg.det(A)
    
    if detA == 0:
        return -1
    
    detlist = []
    for i in range(A.shape[1]):
        tempA = A.copy()
        tempA[:, i] = b   # replace the *i-th column*, not row
        detlist.append(np.linalg.det(tempA))
    
    return [d / detA for d in detlist]
