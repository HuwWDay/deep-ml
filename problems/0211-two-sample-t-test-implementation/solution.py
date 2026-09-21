import numpy as np
from scipy import stats

def two_sample_t_test(sample1: list[float], sample2: list[float], 
                      alpha: float = 0.05) -> dict:
    """
    Perform a two-sample independent t-test (Welch's t-test).
    """
    out = {}

    # Means
    sample1mean = sum(sample1) / len(sample1)
    sample2mean = sum(sample2) / len(sample2)

    # Variances (unbiased)
    sample1var = sum((x - sample1mean)**2 for x in sample1) / (len(sample1) - 1)
    sample2var = sum((x - sample2mean)**2 for x in sample2) / (len(sample2) - 1)

    # Standard error of difference
    SE = np.sqrt(sample1var / len(sample1) + sample2var / len(sample2))

    # t-statistic
    t = (sample1mean - sample2mean) / SE
    out["t_statistic"] = t

    # Welch–Satterthwaite degrees of freedom
    numerator = (sample1var/len(sample1) + sample2var/len(sample2))**2
    denominator = (sample1var**2 / (len(sample1)**2 * (len(sample1) - 1))) + \
                  (sample2var**2