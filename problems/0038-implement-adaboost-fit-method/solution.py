import numpy as np

def adaboost_fit(X, y, n_clf):
    n_samples, n_features = X.shape
    w = np.full(n_samples, 1 / n_samples)
    clfs = []

    for _ in range(n_clf):
        clf = {}
        min_error = float('inf')

        for feature_i in range(n_features):
            X_column = X[:, feature_i]
            sorted_vals = np.sort(np.unique(X_column))
            # Use midpoints between consecutive values as thresholds
            thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0

            for threshold in thresholds:
                for polarity in [1, -1]:
                    pred = np.ones(n_samples)
                    if polarity == 1:
                        pred[X_column < threshold] = -1
                    else:
                        pred[X_column > threshold] = -1

                    error = np.sum(w * (pred != y))
                    current_polarity = polarity

                    if error > 0.5:
                        error =