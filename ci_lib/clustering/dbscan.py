"""DBSCAN clustering algorithm implemented with NumPy."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class DBSCAN:
    """Density-Based Spatial Clustering of Applications with Noise.

    Parameters
    ----------
    eps : float, optional
        Maximum distance between two samples to be considered neighbours
        (default 0.5).
    min_samples : int, optional
        Minimum samples in a neighbourhood to form a dense region
        (default 5).
    metric : str, optional
        Distance metric. Only ``'euclidean'`` is supported.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels. Noisy samples receive label ``-1``.
    core_sample_indices_ : ndarray of shape (n_core_samples,)
        Indices of core samples.

    References
    ----------
    .. [1] Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996).
           A density-based algorithm for discovering clusters. KDD.
    """

    def __init__(
        self, eps: float = 0.5, min_samples: int = 5, metric: str = "euclidean"
    ) -> None:
        if metric != "euclidean":
            raise ValueError(
                f"Unsupported metric '{metric}'. Only 'euclidean' is supported."
            )
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

        self.labels_: npt.NDArray[np.intp] | None = None
        self.core_sample_indices_: npt.NDArray[np.intp] | None = None

    def fit(self, X: npt.NDArray[np.float64]) -> DBSCAN:
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        dist_matrix = self._pairwise_distances(X)

        neighbours = [
            np.where(dist_matrix[i] <= self.eps)[0] for i in range(n_samples)
        ]

        core_mask = np.array(
            [len(neighbours[i]) >= self.min_samples for i in range(n_samples)],
            dtype=bool,
        )
        self.core_sample_indices_ = np.where(core_mask)[0]

        labels = -np.ones(n_samples, dtype=np.intp)
        cluster_id = 0

        for i in range(n_samples):
            if labels[i] != -1 or not core_mask[i]:
                continue

            self._expand_cluster(labels, i, cluster_id, neighbours, core_mask)
            cluster_id += 1

        self.labels_ = labels
        return self

    def fit_predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
        self.fit(X)
        return self.labels_

    @property
    def n_clusters_(self) -> int:
        if self.labels_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        unique = np.unique(self.labels_)
        return int(np.sum(unique != -1))

    @staticmethod
    def _pairwise_distances(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        sq_norms = np.sum(X**2, axis=1)
        dot = X @ X.T
        dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * dot
        np.maximum(dist_sq, 0.0, out=dist_sq)
        return np.sqrt(dist_sq)

    @staticmethod
    def _expand_cluster(
        labels: npt.NDArray[np.intp],
        seed_idx: int,
        cluster_id: int,
        neighbours: list[npt.NDArray[np.intp]],
        core_mask: npt.NDArray[np.bool_],
    ) -> None:
        queue = [seed_idx]
        labels[seed_idx] = cluster_id

        while queue:
            current = queue.pop(0)
            for nb in neighbours[current]:
                if labels[nb] == -1:
                    labels[nb] = cluster_id
                    if core_mask[nb]:
                        queue.append(nb)
