def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	num_features = len(vectors)
    covvar = [[0]*num_features for i in range(num_features)]
    for i in range(num_features):
        for j in range(num_features):
            mean1 = sum(vectors[i])/len(vectors[i])
            mean2 = sum(vectors[j])/len(vectors[j])

            cov = 0
            for k in range(len(vectors[i])):
                cov += (vectors[i][k]-mean1)*(vectors[j][k]-mean2)
            covvar[i][j] = cov/2
    return covvar
            

    
    
        
	