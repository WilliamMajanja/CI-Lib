"""Activation functions for neural networks.

Provides forward and derivative forms of common activation functions,
plus a registry lookup via ``get_activation``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt


def sigmoid(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    s = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    return s


def sigmoid_derivative(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.tanh(x)


def tanh_derivative(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 1.0 - np.tanh(x) ** 2


def relu(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.maximum(0, x)


def relu_derivative(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return (x > 0).astype(np.float64)


def leaky_relu(x: npt.NDArray[np.float64], alpha: float = 0.01) -> npt.NDArray[np.float64]:
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x: npt.NDArray[np.float64], alpha: float = 0.01) -> npt.NDArray[np.float64]:
    return np.where(x > 0, 1.0, alpha)


def softmax(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


ActivationFn = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
DerivativeFn = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None


def get_activation(name: str) -> tuple[ActivationFn, DerivativeFn]:
    registry: dict[str, tuple[ActivationFn, DerivativeFn]] = {
        "sigmoid": (sigmoid, sigmoid_derivative),
        "tanh": (tanh, tanh_derivative),
        "relu": (relu, relu_derivative),
        "leaky_relu": (leaky_relu, leaky_relu_derivative),
        "softmax": (softmax, None),
    }
    if name not in registry:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Choose from {list(registry.keys())}."
        )
    return registry[name]
