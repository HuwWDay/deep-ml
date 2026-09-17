import numpy as np


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with an intercept.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.

    Returns:
        np.ndarray: (p + 1,) coefficients, intercept first.
    """
    n = X.shape[0]
    X_design = np.hstack([np.ones((n, 1)), X])
    # np.linalg.lstsq solves for beta in (X_design.T @ X_design) @ beta = X_design.T @ y
    beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    return beta


def omitted_variable_bias(X: np.ndarray, y: np.ndarray, omit_idx: int) -> tuple:
    """Return (full_kept, short, bias) for a two-column X.

    Args:
        X (np.ndarray): (n, 2) design matrix.
        y (np.ndarray): (n,) target vector.
        omit_idx (int): Index of variable to omit (0 or 1).

    Returns:
        tuple: (full_kept, short, bias)
            - full_kept: coefficient of the kept variable in the full model.
            - short: coefficient of the kept variable in the short model.
            - bias: short - full_kept.
    """
    kept_idx = 1 - omit_idx

    # Full model: includes intercept, X[:, 0], and X[:, 1]
    beta_full = ols(X, y)
    # beta_full[0] is intercept, beta_full[1] is col 0, beta_full[2] is col 1
    full_kept = beta_full[kept_idx + 1]

    # Short model: includes intercept and only the kept column
    X_short = X[:, [kept_idx]]
    beta_short = ols(X_short, y)
    # beta_short[0] is intercept, beta_short[1] is the kept variable's coefficient
    short = beta_short[1]

    bias = short - full_kept
    return float(full_kept), float(short), float(bias)