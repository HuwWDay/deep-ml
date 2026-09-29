import numpy as np


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Poisson deviance, using the convention 0 * log(0) = 0."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)

    # Guard the log computation to avoid RuntimeWarning on the unselected branch
    y_safe = np.where(y > 0, y, 1.0)
    term1 = np.where(y > 0, y * np.log(y_safe / mu), 0.0)
    term2 = y - mu

    return float(2.0 * np.sum(term1 - term2))


def dispersion_ratio(y: np.ndarray, mu: np.ndarray, n_params: int) -> float:
    """Pearson chi-square divided by (n - n_params)."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)

    pearson_chi2 = np.sum((y - mu) ** 2 / mu)
    dof = len(y) - n_params

    return float(pearson_chi2 / dof)