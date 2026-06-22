"""K-Means clustering algorithm implemented with NumPy."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class KMeans:
    """K-Means clustering with k-means++ initialisation.

    Parameters
    ----------
    n_clusters : int, optional
        Number of clusters to form (default 3).
    max_iter : int, optional
        Maximum iterations for a single run (default 300).
    tol : float, optional
        Relative tolerance on inertia for convergence (default 1e-4).
    init : {'kmeans++', 'random'}, optional
        Centroid initialisation method (default 'kmeans++').
    seed : int or None, optional
        Random seed for reproducibility.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centres.
    labels_ : ndarray of shape (n_samples,)
        Label of each sample.
    inertia_ : float
        Sum of squared distances to nearest cluster centre.

    References
    ----------
    .. [1] Lloyd, S. P. (1982). Least squares quantization in PCM.
           IEEE Transactions on Information Theory.
    .. [2] Arthur, D., & Vassilvitskii, S. (2007). k-means++: The
           advantages of careful seeding. SODA.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        init: str = "kmeans++",
        seed: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.seed = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)

        self.cluster_centers_: npt.NDArray[np.float64] | None = None
        self.labels_: npt.NDArray[np.intp] | None = None
        self.inertia_: float | None = None

    def fit(self, X: npt.NDArray[np.float64]) -> KMeans:
        X = np.asarray(X, dtype=np.float64)
        centroids = self._init_centroids(X)

        for _ in range(self.max_iter):
            labels = self._assign_labels(X, centroids)
            new_centroids = self._compute_centroids(X, labels)
            shift = np.sum((new_centroids - centroids) ** 2)
            centroids = new_centroids
            if shift < self.tol:
                break

        self.cluster_centers_ = centroids
        self.labels_ = self._assign_labels(X, centroids)
        self.inertia_ = float(self._compute_inertia(X, centroids, self.labels_))
        return self

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
        if self.cluster_centers_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return self._assign_labels(X, self.cluster_centers_)

    def fit_predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
        self.fit(X)
        return self.labels_

    def silhouette_score(self, X: npt.NDArray[np.float64]) -> float:
        if self.labels_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X = np.asarray(X, dtype=np.float64)
        labels = self.labels_
        unique_labels = np.unique(labels)

        if len(unique_labels) < 2:
            raise ValueError(
                "Silhouette score requires at least 2 clusters, "
                f"got {len(unique_labels)}."
            )

        n_samples = X.shape[0]
        scores = np.zeros(n_samples)

        for i in range(n_samples):
            own_mask = labels == labels[i]
            own_cluster_size = int(np.sum(own_mask))

            if own_cluster_size == 1:
                a_i = 0.0
            else:
                dists = np.linalg.norm(X[own_mask] - X[i], axis=1)
                a_i = np.sum(dists) / (own_cluster_size - 1)

            b_i = np.inf
            for label in unique_labels:
                if label == labels[i]:
                    continue
                other_mask = labels == label
                dists = np.linalg.norm(X[other_mask] - X[i], axis=1)
                mean_dist = np.mean(dists)
                if mean_dist < b_i:
                    b_i = mean_dist

            denom = max(a_i, b_i)
            scores[i] = (b_i - a_i) / denom if denom > 0 else 0.0

        return float(np.mean(scores))

    def _init_centroids(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        n_samples = X.shape[0]

        if self.init == "random":
            indices = self._rng.choice(n_samples, size=self.n_clusters, replace=False)
            return X[indices].copy()

        if self.init == "kmeans++":
            centroids = np.empty((self.n_clusters, X.shape[1]), dtype=np.float64)
            idx = int(self._rng.integers(0, n_samples))
            centroids[0] = X[idx]

            for k in range(1, self.n_clusters):
                dists = np.min(
                    np.linalg.norm(
                        X[:, np.newaxis, :] - centroids[np.newaxis, :k, :], axis=2
                    ),
                    axis=1,
                )
                probs = dists**2
                probs_sum = np.sum(probs)
                if probs_sum > 0:
                    probs = probs / probs_sum
                else:
                    probs = np.ones(n_samples) / n_samples
                next_idx = int(self._rng.choice(n_samples, p=probs))
                centroids[k] = X[next_idx]

            return centroids

        raise ValueError(
            f"Unknown init method '{self.init}'. Use 'kmeans++' or 'random'."
        )

    @staticmethod
    def _assign_labels(
        X: npt.NDArray[np.float64], centroids: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.intp]:
        dists = np.linalg.norm(
            X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
        )
        return np.argmin(dists, axis=1).astype(np.intp)

    def _compute_centroids(
        self, X: npt.NDArray[np.float64], labels: npt.NDArray[np.intp]
    ) -> npt.NDArray[np.float64]:
        centroids = np.empty((self.n_clusters, X.shape[1]), dtype=np.float64)
        for k in range(self.n_clusters):
            members = X[labels == k]
            if len(members) > 0:
                centroids[k] = members.mean(axis=0)
            else:
                centroids[k] = X[int(self._rng.integers(0, X.shape[0]))]
        return centroids

    @staticmethod
    def _compute_inertia(
        X: npt.NDArray[np.float64],
        centroids: npt.NDArray[np.float64],
        labels: npt.NDArray[np.intp],
    ) -> float:
        return float(np.sum((X - centroids[labels]) ** 2))
