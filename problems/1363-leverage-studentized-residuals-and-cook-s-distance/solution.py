import numpy as np


def influence_measures(X: np.ndarray, y: np.ndarray) -> tuple:
    """Leverage, studentized residuals and Cook's distance for every observation.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.

    Returns:
        tuple: (leverage, studentized, cooks), each a length-n array.
    """
    n, p = X.shape
    k = p + 1

    # 1. Augment design matrix with intercept column of ones: D = [1, X]
    D = np.column_stack([np.ones(n), X])

    # 2. Compute OLS coefficients beta = (D^T D)^(-1) D^T y
    # np.linalg.pinv or np.linalg.lstsq can also be used, but standard normal equations:
    inv_DTD = np.linalg.inv(D.T @ D)
    beta = inv_DTD @ D.T @ y

    # 3. Leverage values h_i = H_ii
    # Instead of full (n, n) matrix multiplication D @ inv_DTD @ D.T,
    # compute diagonal elements efficiently: sum((D @ inv_DTD) * D, axis=1)
    leverage = np.sum((D @ inv_DTD) * D, axis=1)

    # 4. Residuals and residual variance estimate (sigma^2)
    y_pred = D @ beta
    e = y - y_pred
    rss = np.sum(e**2)
    sigma2 = rss / (n - k)

    # 5. Internally studentized residuals: r_i = e_i / (sigma * sqrt(1 - h_i))
    se_residuals = np.sqrt(sigma2 * (1.0 - leverage))
    studentized = e / se_residuals

    # 6. Cook's distance: D_i = (r_i^2 / k) * (h_i / (1 - h_i))
    cooks = (studentized**2 / k) * (leverage / (1.0 - leverage))

    return leverage, studentized, cooks