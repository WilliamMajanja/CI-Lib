"""Tests for ci_lib.clustering (KMeans + DBSCAN)."""

import numpy as np
import pytest

from ci_lib.clustering.kmeans import KMeans
from ci_lib.clustering.dbscan import DBSCAN


def _make_blobs(seed=42):
    """Generate 3 well-separated 2-D clusters."""
    rng = np.random.default_rng(seed)
    c1 = rng.normal(loc=[0, 0], scale=0.3, size=(30, 2))
    c2 = rng.normal(loc=[5, 5], scale=0.3, size=(30, 2))
    c3 = rng.normal(loc=[10, 0], scale=0.3, size=(30, 2))
    return np.vstack([c1, c2, c3])


# ------------------------------------------------------------------
# KMeans
# ------------------------------------------------------------------

class TestKMeans:
    def test_labels_shape_and_range(self):
        X = _make_blobs(seed=42)
        km = KMeans(n_clusters=3, seed=42)
        km.fit(X)
        assert km.labels_.shape == (X.shape[0],)
        assert set(km.labels_).issubset({0, 1, 2})

    def test_cluster_centers_shape(self):
        X = _make_blobs(seed=42)
        km = KMeans(n_clusters=3, seed=42)
        km.fit(X)
        assert km.cluster_centers_.shape == (3, 2)

    def test_inertia_positive(self):
        X = _make_blobs(seed=42)
        km = KMeans(n_clusters=3, seed=42)
        km.fit(X)
        assert km.inertia_ > 0.0

    def test_predict_matches_fit_predict(self):
        X = _make_blobs(seed=42)
        km = KMeans(n_clusters=3, seed=42)
        labels_fp = km.fit_predict(X)
        labels_p = km.predict(X)
        np.testing.assert_array_equal(labels_fp, labels_p)

    def test_predict_before_fit_raises(self):
        km = KMeans(n_clusters=3, seed=42)
        with pytest.raises(RuntimeError):
            km.predict(np.array([[0, 0]]))

    @pytest.mark.parametrize("init", ["kmeans++", "random"])
    def test_init_methods(self, init):
        X = _make_blobs(seed=42)
        km = KMeans(n_clusters=3, init=init, seed=42)
        km.fit(X)
        assert km.labels_.shape == (X.shape[0],)


# ------------------------------------------------------------------
# DBSCAN
# ------------------------------------------------------------------

class TestDBSCAN:
    def test_detects_noise(self):
        rng = np.random.default_rng(42)
        cluster = rng.normal(loc=[0, 0], scale=0.1, size=(30, 2))
        noise = np.array([[10.0, 10.0], [12.0, 12.0]])
        X = np.vstack([cluster, noise])
        db = DBSCAN(eps=0.5, min_samples=5)
        labels = db.fit_predict(X)
        assert -1 in labels

    def test_n_clusters_property(self):
        X = _make_blobs(seed=42)
        db = DBSCAN(eps=1.0, min_samples=3)
        db.fit(X)
        assert db.n_clusters_ >= 1
