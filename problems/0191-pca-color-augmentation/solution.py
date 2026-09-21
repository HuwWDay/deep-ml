import numpy as np

def pca_color_augmentation(image: np.ndarray, alpha: np.ndarray = None) -> np.ndarray:
    """
    Apply PCA color augmentation to an RGB image.
    
    Args:
        image: Input image of shape (H, W, 3).
        alpha: Random coefficients of shape (3,). 
        
    Returns:
        Augmented image of shape (H, W, 3) as float32.
    """
    # 1. Convert to float and normalized range [0, 1] for stability
    # This prevents covariance values from exploding
    img_float = image.astype(np.float32) / 255.0
    
    # 2. Reshape to (N, 3) to treat pixels as data points
    flattened_img = img_float.reshape(-1, 3)
    
    # 3. Compute Covariance Matrix
    # rowvar=False because rows are observations (pixels), columns are variables (R,G,B)
    covariance_matrix = np.cov(flattened_img, rowvar=False)
    
    # 4. Eigen Decomposition
    # eigh returns eigenvalues in ascending order
    eigen_values, eigen_vectors = np.linalg.eigh(covariance_