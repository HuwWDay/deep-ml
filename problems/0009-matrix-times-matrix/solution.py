def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    
    if len(a[0]) != len(b):
        return -1 
    else:
        # Initialize c with zeros
        c = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]

        # Perform matrix multiplication
        # c_ij = sum_k a_ik * b_kj
        for i in range(len(a)):
            for j in range(len(b[0])):
                for k in range(len(b)):
                    c[i][j] += a[i][k] * b[k][j]
        return c