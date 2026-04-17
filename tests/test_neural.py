"""Tests for ci_lib.neural (activations + FeedForwardNetwork)."""

import numpy as np
import pytest

from ci_lib.neural.activations import (
    sigmoid,
    sigmoid_derivative,
    tanh,
    relu,
    leaky_relu,
    softmax,
    get_activation,
)
from ci_lib.neural.feedforward import FeedForwardNetwork


# ------------------------------------------------------------------
# Activation function tests
# ------------------------------------------------------------------

class TestSigmoid:
    def test_sigmoid_at_zero(self):
        assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_sigmoid_range(self):
        x = np.linspace(-10, 10, 50)
        out = sigmoid(x)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_sigmoid_derivative_at_zero(self):
        assert sigmoid_derivative(np.array([0.0]))[0] == pytest.approx(0.25)


class TestRelu:
    def test_relu_negative(self):
        assert relu(np.array([-1.0]))[0] == 0.0

    def test_relu_positive(self):
        assert relu(np.array([3.0]))[0] == 3.0

    def test_relu_zero(self):
        assert relu(np.array([0.0]))[0] == 0.0


class TestLeakyRelu:
    def test_leaky_relu_negative(self):
        result = leaky_relu(np.array([-10.0]), alpha=0.01)
        assert result[0] == pytest.approx(-0.1)


class TestTanh:
    def test_tanh_at_zero(self):
        assert tanh(np.array([0.0]))[0] == pytest.approx(0.0)


class TestSoftmax:
    def test_softmax_sums_to_one(self):
        x = np.array([[1.0, 2.0, 3.0]])
        out = softmax(x)
        assert out.sum() == pytest.approx(1.0)


class TestGetActivation:
    @pytest.mark.parametrize("name", ["sigmoid", "tanh", "relu", "leaky_relu", "softmax"])
    def test_valid_names(self, name):
        fn, deriv = get_activation(name)
        assert callable(fn)

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation("nonexistent")


# ------------------------------------------------------------------
# FeedForwardNetwork tests
# ------------------------------------------------------------------

class TestFeedForwardNetwork:
    def test_predict_shape(self):
        nn = FeedForwardNetwork([2, 4, 1], seed=42)
        X = np.array([[0, 0], [1, 1]])
        out = nn.predict(X)
        assert out.shape == (2, 1)

    def test_fit_xor_reduces_loss(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        nn = FeedForwardNetwork([2, 8, 1], activation="sigmoid",
                                learning_rate=0.5, seed=42)
        nn.fit(X, y, epochs=2000)
        loss = nn.score(X, y)
        assert loss < 0.25

    @pytest.mark.parametrize("activation", ["sigmoid", "tanh", "relu"])
    def test_activations_run(self, activation):
        nn = FeedForwardNetwork([2, 3, 1], activation=activation, seed=42)
        out = nn.predict(np.array([[1.0, 2.0]]))
        assert out.shape == (1, 1)

    def test_invalid_layer_sizes_not_list(self):
        with pytest.raises(TypeError, match="layer_sizes must be a list"):
            FeedForwardNetwork((2, 1))

    def test_invalid_layer_sizes_too_short(self):
        with pytest.raises(ValueError, match="at least two elements"):
            FeedForwardNetwork([5])

    def test_invalid_layer_sizes_bad_type(self):
        with pytest.raises(TypeError, match="must be an integer"):
            FeedForwardNetwork([2, 3.5, 1])

    def test_invalid_layer_sizes_zero(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            FeedForwardNetwork([2, 0, 1])
