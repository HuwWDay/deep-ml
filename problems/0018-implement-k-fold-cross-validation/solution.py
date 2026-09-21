import numpy as np

def k_fold_cross_validation(X: np.ndarray, y: np.ndarray, k=5, shuffle=True):
    """
    Implement k-fold cross-validation by returning train-test indices.
    """
    n_samples = len(X)
    indices = np.arange(n_samples)

    # Shuffle indices if requested
    if shuffle:
        np.random.shuffle(indices)

    # Split indices into k folds
    folds = np.array_split(indices, k)

    # Create train/test splits
    splits = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, test_idx))

    return splits