"""DBSCAN clustering algorithm implemented with NumPy."""

import numpy as np


class DBSCAN:
    """Density-Based Spatial Clustering of Applications with Noise.

    Parameters
    ----------
    eps : float, optional
        Maximum distance between two samples for them to be considered
        neighbours. Default is 0.5.
    min_samples : int, optional
        Minimum number of samples in a neighbourhood (including the point
        itself) to form a dense region. Default is 5.
    metric : str, optional
        Distance metric. Only ``'euclidean'`` is supported. Default is
        ``'euclidean'``.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels. Noisy samples receive the label ``-1``.
    core_sample_indices_ : ndarray
        Indices of core samples.
    """

    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        if metric != 'euclidean':
            raise ValueError(
                f"Unsupported metric '{metric}'. Only 'euclidean' is supported."
            )
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

        self.labels_ = None
        self.core_sample_indices_ = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X):
        """Perform DBSCAN clustering.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        # Pairwise distance matrix
        dist_matrix = self._pairwise_distances(X)

        # Determine neighbours for every point
        neighbours = [
            np.where(dist_matrix[i] <= self.eps)[0] for i in range(n_samples)
        ]

        # Identify core samples
        core_mask = np.array(
            [len(neighbours[i]) >= self.min_samples for i in range(n_samples)]
        )
        self.core_sample_indices_ = np.where(core_mask)[0]

        # Assign labels via BFS expansion
        labels = -np.ones(n_samples, dtype=np.intp)
        cluster_id = 0

        for i in range(n_samples):
            if labels[i] != -1 or not core_mask[i]:
                continue

            # Expand cluster from core point *i*
            self._expand_cluster(
                labels, i, cluster_id, neighbours, core_mask
            )
            cluster_id += 1

        self.labels_ = labels
        return self

    def fit_predict(self, X):
        """Fit the model and return cluster labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels (``-1`` for noise).
        """
        self.fit(X)
        return self.labels_

    @property
    def n_clusters_(self):
        """Number of clusters found (excluding noise).

        Returns
        -------
        int
            Count of unique labels that are not ``-1``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.labels_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        unique = np.unique(self.labels_)
        return int(np.sum(unique != -1))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pairwise_distances(X):
        """Compute the pairwise Euclidean distance matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data points.

        Returns
        -------
        dist : ndarray of shape (n_samples, n_samples)
            Symmetric distance matrix.
        """
        sq_norms = np.sum(X ** 2, axis=1)
        dot = X @ X.T
        dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * dot
        np.maximum(dist_sq, 0.0, out=dist_sq)
        return np.sqrt(dist_sq)

    @staticmethod
    def _expand_cluster(labels, seed_idx, cluster_id, neighbours, core_mask):
        """Expand a cluster using breadth-first search.

        Parameters
        ----------
        labels : ndarray of shape (n_samples,)
            Label array (modified in-place).
        seed_idx : int
            Index of the starting core point.
        cluster_id : int
            Cluster identifier to assign.
        neighbours : list of ndarray
            Pre-computed neighbour lists for every sample.
        core_mask : ndarray of shape (n_samples,)
            Boolean mask indicating core points.
        """
        queue = [seed_idx]
        labels[seed_idx] = cluster_id

        while queue:
            current = queue.pop(0)
            for nb in neighbours[current]:
                if labels[nb] == -1:
                    labels[nb] = cluster_id
                    if core_mask[nb]:
                        queue.append(nb)
