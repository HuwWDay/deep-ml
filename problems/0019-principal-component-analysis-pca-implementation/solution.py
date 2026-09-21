import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    # Standardize the data
    DataNormed = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    # Compute covariance matrix (columns = features)
    Cov = np.cov(DataNormed, rowvar=False)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(Cov)
    
    # Sort eigenvalues (and eigenvectors) in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top-k eigenvectors
    principal_components = eigenvectors[:, :k]
    
    
    
    return np.round(principal_components, 4)
