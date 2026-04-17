"""Tests for ci_lib.utils (preprocessing, metrics, benchmarks)."""

import numpy as np
import pytest

from ci_lib.utils.preprocessing import normalize, train_test_split, one_hot_encode
from ci_lib.utils.metrics import accuracy, mse, rmse, mae, r_squared
from ci_lib.utils.benchmarks import sphere, rosenbrock, rastrigin


# ------------------------------------------------------------------
# Preprocessing: normalize
# ------------------------------------------------------------------

class TestNormalize:
    def test_minmax_output_range(self):
        rng = np.random.default_rng(42)
        X = rng.random((50, 3)) * 100
        out = normalize(X, method="minmax")
        assert out.min() >= 0.0 - 1e-12
        assert out.max() <= 1.0 + 1e-12

    def test_zscore_mean_std(self):
        rng = np.random.default_rng(42)
        X = rng.random((100, 4)) * 50 + 10
        out = normalize(X, method="zscore")
        assert np.abs(out.mean(axis=0)).max() < 1e-10
        assert np.allclose(out.std(axis=0), 1.0, atol=1e-10)


# ------------------------------------------------------------------
# Preprocessing: train_test_split
# ------------------------------------------------------------------

class TestTrainTestSplit:
    def test_sizes(self):
        rng = np.random.default_rng(42)
        X = rng.random((100, 5))
        y = rng.integers(0, 3, size=100)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                              test_size=0.2,
                                                              seed=42)
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert y_train.shape[0] == 80
        assert y_test.shape[0] == 20


# ------------------------------------------------------------------
# Preprocessing: one_hot_encode
# ------------------------------------------------------------------

class TestOneHotEncode:
    def test_shape(self):
        y = np.array([0, 1, 2, 1, 0])
        encoded = one_hot_encode(y)
        assert encoded.shape == (5, 3)

    def test_row_sums_to_one(self):
        y = np.array([0, 1, 2])
        encoded = one_hot_encode(y)
        np.testing.assert_array_almost_equal(encoded.sum(axis=1), 1.0)


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

class TestAccuracy:
    def test_perfect(self):
        assert accuracy([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_half(self):
        assert accuracy([1, 1, 0, 0], [1, 0, 0, 1]) == pytest.approx(0.5)


class TestMSE:
    def test_zero_error(self):
        assert mse([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_known_value(self):
        assert mse([0, 0], [1, 1]) == pytest.approx(1.0)


class TestRMSE:
    def test_known_value(self):
        assert rmse([0, 0], [1, 1]) == pytest.approx(1.0)


class TestMAE:
    def test_known_value(self):
        assert mae([0, 0, 0], [1, -1, 2]) == pytest.approx(4.0 / 3.0)


class TestRSquared:
    def test_perfect(self):
        assert r_squared([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_bad(self):
        val = r_squared([1, 2, 3], [3, 2, 1])
        assert val < 1.0


# ------------------------------------------------------------------
# Benchmarks
# ------------------------------------------------------------------

class TestBenchmarks:
    def test_sphere_at_origin(self):
        assert sphere(np.array([0.0, 0.0])) == pytest.approx(0.0)

    def test_rosenbrock_at_ones(self):
        assert rosenbrock(np.array([1.0, 1.0])) == pytest.approx(0.0)

    def test_rosenbrock_not_at_origin(self):
        assert rosenbrock(np.array([0.0, 0.0])) > 0.0

    def test_rastrigin_at_origin(self):
        assert rastrigin(np.array([0.0, 0.0])) == pytest.approx(0.0)
