"""K-Means clustering algorithm implemented with NumPy."""

import numpy as np


class KMeans:
    """K-Means clustering.

    Parameters
    ----------
    n_clusters : int, optional
        Number of clusters to form. Default is 3.
    max_iter : int, optional
        Maximum number of iterations for a single run. Default is 300.
    tol : float, optional
        Relative tolerance with respect to inertia to declare convergence.
        Default is 1e-4.
    init : str, optional
        Centroid initialisation method. ``'kmeans++'`` (default) or
        ``'random'``.
    seed : int or None, optional
        Seed for the random number generator. Default is None.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centres.
    labels_ : ndarray of shape (n_samples,)
        Label of each sample.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster centre.
    """

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4,
                 init='kmeans++', seed=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X):
        """Compute K-Means clustering.

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

    def predict(self, X):
        """Predict the closest cluster for each sample in *X*.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return self._assign_labels(X, self.cluster_centers_)

    def fit_predict(self, X):
        """Fit the model and return cluster labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        self.fit(X)
        return self.labels_

    def silhouette_score(self, X):
        """Compute the mean silhouette score over all samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data that was used for fitting.

        Returns
        -------
        score : float
            Mean silhouette coefficient in the range ``[-1, 1]``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        ValueError
            If fewer than two clusters were found.
        """
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

            # Mean intra-cluster distance (a)
            if own_cluster_size == 1:
                a_i = 0.0
            else:
                dists = np.linalg.norm(X[own_mask] - X[i], axis=1)
                a_i = np.sum(dists) / (own_cluster_size - 1)

            # Mean nearest-cluster distance (b)
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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_centroids(self, X):
        """Initialise cluster centroids.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        centroids : ndarray of shape (n_clusters, n_features)
            Initial centroid positions.
        """
        n_samples = X.shape[0]

        if self.init == 'random':
            indices = self._rng.choice(n_samples, size=self.n_clusters,
                                       replace=False)
            return X[indices].copy()

        if self.init == 'kmeans++':
            centroids = np.empty((self.n_clusters, X.shape[1]),
                                 dtype=np.float64)
            idx = int(self._rng.integers(0, n_samples))
            centroids[0] = X[idx]

            for k in range(1, self.n_clusters):
                dists = np.min(
                    np.linalg.norm(
                        X[:, np.newaxis, :] - centroids[np.newaxis, :k, :],
                        axis=2,
                    ),
                    axis=1,
                )
                probs = dists ** 2
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
    def _assign_labels(X, centroids):
        """Return the index of the nearest centroid for each sample.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data points.
        centroids : ndarray of shape (n_clusters, n_features)
            Current centroid positions.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster assignments.
        """
        dists = np.linalg.norm(
            X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
        )
        return np.argmin(dists, axis=1)

    def _compute_centroids(self, X, labels):
        """Recompute centroids as the mean of assigned samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data points.
        labels : ndarray of shape (n_samples,)
            Current cluster assignments.

        Returns
        -------
        centroids : ndarray of shape (n_clusters, n_features)
            Updated centroid positions.
        """
        centroids = np.empty((self.n_clusters, X.shape[1]), dtype=np.float64)
        for k in range(self.n_clusters):
            members = X[labels == k]
            if len(members) > 0:
                centroids[k] = members.mean(axis=0)
            else:
                centroids[k] = X[int(self._rng.integers(0, X.shape[0]))]
        return centroids

    @staticmethod
    def _compute_inertia(X, centroids, labels):
        """Compute the sum of squared distances to the nearest centroid.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data points.
        centroids : ndarray of shape (n_clusters, n_features)
            Centroid positions.
        labels : ndarray of shape (n_samples,)
            Cluster assignments.

        Returns
        -------
        inertia : float
            Total within-cluster sum of squares.
        """
        return np.sum((X - centroids[labels]) ** 2)
