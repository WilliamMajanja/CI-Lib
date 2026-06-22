"""Feed-forward neural network trained with back-propagation.

Uses only NumPy. Supports mini-batch SGD, configurable activations,
and Xavier weight initialisation.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt

from ci_lib.neural.activations import get_activation, ActivationFn, DerivativeFn


class FeedForwardNetwork:
    """A fully-connected feed-forward neural network.

    Parameters
    ----------
    layer_sizes : list of int
        Number of neurons in each layer, including input and output.
    activation : str, optional
        Name of the activation function (default ``'sigmoid'``).
    learning_rate : float, optional
        Step size for gradient descent (default ``0.01``).
    seed : int or None, optional
        Random seed for reproducibility.

    References
    ----------
    .. [1] Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).
           Learning representations by back-propagating errors. Nature.
    .. [2] Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of
           training deep feedforward neural networks. AISTATS.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "sigmoid",
        learning_rate: float = 0.01,
        seed: int | None = None,
    ) -> None:
        self._validate_layer_sizes(layer_sizes)
        self.layer_sizes: list[int] = list(layer_sizes)
        self.learning_rate: float = learning_rate
        self.rng: np.random.Generator = np.random.default_rng(seed)

        act_fn, act_deriv = get_activation(activation)
        self.activation: ActivationFn = act_fn
        self.activation_derivative: DerivativeFn = act_deriv
        self.activation_name: str = activation

        self.weights: list[npt.NDArray[np.float64]] = []
        self.biases: list[npt.NDArray[np.float64]] = []
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            w = self.rng.uniform(-limit, limit, size=(fan_in, fan_out))
            b = np.zeros((1, fan_out), dtype=np.float64)
            self.weights.append(w)
            self.biases.append(b)

        self._activations: list[npt.NDArray[np.float64]] = []
        self._pre_activations: list[npt.NDArray[np.float64]] = []

    @staticmethod
    def _validate_layer_sizes(layer_sizes: list[int]) -> None:
        if not isinstance(layer_sizes, list):
            raise TypeError("layer_sizes must be a list of integers.")
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least two elements.")
        for size in layer_sizes:
            if not isinstance(size, int):
                raise TypeError(
                    f"Each layer size must be an integer, got {type(size).__name__}."
                )
            if size < 1:
                raise ValueError(f"Each layer size must be >= 1, got {size}.")

    def forward(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        X = np.atleast_2d(X)
        self._activations = [X]
        self._pre_activations = []

        current = X
        for w, b in zip(self.weights, self.biases):
            z = current @ w + b
            self._pre_activations.append(z)
            current = self.activation(z)
            self._activations.append(current)

        return current

    def backward(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
        X = np.atleast_2d(X)
        y = np.atleast_2d(y)
        m = X.shape[0]

        output = self.forward(X)

        error = output - y
        if self.activation_derivative is not None:
            delta = error * self.activation_derivative(self._pre_activations[-1])
        else:
            delta = error

        deltas = [delta]

        for i in range(len(self.weights) - 2, -1, -1):
            error = deltas[-1] @ self.weights[i + 1].T
            if self.activation_derivative is not None:
                delta = error * self.activation_derivative(self._pre_activations[i])
            else:
                delta = error
            deltas.append(delta)

        deltas.reverse()

        for i in range(len(self.weights)):
            grad_w = self._activations[i].T @ deltas[i] / m
            grad_b = np.sum(deltas[i], axis=0, keepdims=True) / m
            self.weights[i] -= self.learning_rate * grad_w
            self.biases[i] -= self.learning_rate * grad_b

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        epochs: int = 1000,
        batch_size: int | None = None,
        verbose: bool = False,
    ) -> FeedForwardNetwork:
        X = np.atleast_2d(X)
        y = np.atleast_2d(y)
        n_samples = X.shape[0]

        for epoch in range(epochs):
            indices = self.rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            if batch_size is None:
                self.backward(X_shuffled, y_shuffled)
            else:
                for start in range(0, n_samples, batch_size):
                    end = start + batch_size
                    self.backward(X_shuffled[start:end], y_shuffled[start:end])

            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                mse_val = self.score(X, y)
                print(f"Epoch {epoch:>5d} | MSE: {mse_val:.6f}")

        return self

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.forward(X)

    def score(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
        y = np.atleast_2d(y)
        predictions = self.forward(X)
        return float(np.mean((predictions - y) ** 2))
