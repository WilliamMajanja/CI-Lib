"""Data preprocessing utilities."""
import numpy as np


def normalize(X, method="minmax"):
    """Normalize features column-wise.

    Parameters
    ----------
    X : array_like
        Input array of shape ``(n_samples, n_features)`` or 1-D.
    method : {'minmax', 'zscore'}, default='minmax'
        ``'minmax'`` scales each feature to [0, 1].
        ``'zscore'`` standardises to zero mean and unit variance.

    Returns
    -------
    numpy.ndarray
        Normalized array with the same shape as *X*.

    Raises
    ------
    ValueError
        If *method* is not ``'minmax'`` or ``'zscore'``.
    """
    X = np.asarray(X, dtype=np.float64)
    if method == "minmax":
        col_min = X.min(axis=0)
        col_max = X.max(axis=0)
        denom = col_max - col_min
        # Avoid division by zero for constant columns
        denom = np.where(denom == 0, 1.0, denom)
        return (X - col_min) / denom
    if method == "zscore":
        col_mean = X.mean(axis=0)
        col_std = X.std(axis=0)
        col_std = np.where(col_std == 0, 1.0, col_std)
        return (X - col_mean) / col_std
    raise ValueError(f"Unknown method '{method}'. Use 'minmax' or 'zscore'.")


def train_test_split(X, y, test_size=0.2, seed=None):
    """Split arrays into random train and test subsets.

    Parameters
    ----------
    X : array_like
        Feature array of shape ``(n_samples, ...)``.
    y : array_like
        Target array of shape ``(n_samples, ...)``.
    test_size : float, default=0.2
        Fraction of samples to reserve for the test set (0, 1).
    seed : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    X_train : numpy.ndarray
    X_test : numpy.ndarray
    y_train : numpy.ndarray
    y_test : numpy.ndarray
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def one_hot_encode(y):
    """One-hot encode an integer label vector.

    Parameters
    ----------
    y : array_like
        1-D array of integer or categorical labels.

    Returns
    -------
    numpy.ndarray
        2-D binary array of shape ``(n_samples, n_classes)``.
    """
    y = np.asarray(y)
    classes = np.unique(y)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    encoded = np.zeros((len(y), len(classes)), dtype=np.float64)
    for i, label in enumerate(y):
        encoded[i, class_to_idx[label]] = 1.0
    return encoded


def label_encode(y):
    """Encode categorical labels as consecutive integers starting from 0.

    Parameters
    ----------
    y : array_like
        1-D array of labels (strings, ints, etc.).

    Returns
    -------
    numpy.ndarray
        1-D integer array of shape ``(n_samples,)``.
    """
    y = np.asarray(y)
    classes, inverse = np.unique(y, return_inverse=True)
    return inverse.astype(np.intp)
