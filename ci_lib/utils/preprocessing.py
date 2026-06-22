"""Data preprocessing utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def normalize(
    X: npt.ArrayLike, method: str = "minmax"
) -> npt.NDArray[np.float64]:
    X = np.asarray(X, dtype=np.float64)
    if method == "minmax":
        col_min = X.min(axis=0)
        col_max = X.max(axis=0)
        denom = col_max - col_min
        denom = np.where(denom == 0, 1.0, denom)
        return (X - col_min) / denom
    if method == "zscore":
        col_mean = X.mean(axis=0)
        col_std = X.std(axis=0)
        col_std = np.where(col_std == 0, 1.0, col_std)
        return (X - col_mean) / col_std
    raise ValueError(f"Unknown method '{method}'. Use 'minmax' or 'zscore'.")


def train_test_split(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    test_size: float = 0.2,
    seed: int | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def one_hot_encode(y: npt.ArrayLike) -> npt.NDArray[np.float64]:
    y = np.asarray(y)
    classes = np.unique(y)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    encoded = np.zeros((len(y), len(classes)), dtype=np.float64)
    for i, label in enumerate(y):
        encoded[i, class_to_idx[label]] = 1.0
    return encoded


def label_encode(y: npt.ArrayLike) -> npt.NDArray[np.intp]:
    y = np.asarray(y)
    _, inverse = np.unique(y, return_inverse=True)
    return inverse.astype(np.intp)
