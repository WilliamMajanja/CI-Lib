"""Activation functions for neural networks.

Provides forward and derivative forms of common activation functions,
plus a registry lookup via ``get_activation``.
"""

import numpy as np


def sigmoid(x):
    """Compute the logistic sigmoid element-wise.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray
        Sigmoid of each element, in the range (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x):
    """Compute the derivative of the sigmoid function element-wise.

    Parameters
    ----------
    x : numpy.ndarray
        Input array (pre-activation values).

    Returns
    -------
    numpy.ndarray
        ``sigmoid(x) * (1 - sigmoid(x))``.
    """
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x):
    """Compute the hyperbolic tangent element-wise.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray
        Tanh of each element, in the range (-1, 1).
    """
    return np.tanh(x)


def tanh_derivative(x):
    """Compute the derivative of tanh element-wise.

    Parameters
    ----------
    x : numpy.ndarray
        Input array (pre-activation values).

    Returns
    -------
    numpy.ndarray
        ``1 - tanh(x)**2``.
    """
    return 1.0 - np.tanh(x) ** 2


def relu(x):
    """Compute the Rectified Linear Unit element-wise.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray
        ``max(0, x)`` for each element.
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """Compute the derivative of ReLU element-wise.

    Parameters
    ----------
    x : numpy.ndarray
        Input array (pre-activation values).

    Returns
    -------
    numpy.ndarray
        1 where ``x > 0``, 0 elsewhere.
    """
    return (x > 0).astype(float)


def leaky_relu(x, alpha=0.01):
    """Compute the Leaky ReLU element-wise.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.
    alpha : float, optional
        Slope for negative values. Default is 0.01.

    Returns
    -------
    numpy.ndarray
        ``x`` where ``x > 0``, ``alpha * x`` elsewhere.
    """
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    """Compute the derivative of Leaky ReLU element-wise.

    Parameters
    ----------
    x : numpy.ndarray
        Input array (pre-activation values).
    alpha : float, optional
        Slope for negative values. Default is 0.01.

    Returns
    -------
    numpy.ndarray
        1 where ``x > 0``, ``alpha`` elsewhere.
    """
    return np.where(x > 0, 1.0, alpha)


def softmax(x):
    """Compute the softmax function row-wise (numerically stable).

    Parameters
    ----------
    x : numpy.ndarray
        Input array of shape ``(n_samples, n_classes)``.

    Returns
    -------
    numpy.ndarray
        Softmax probabilities with the same shape as *x*.

    Notes
    -----
    No standalone derivative is provided because softmax back-propagation
    is typically fused with the cross-entropy loss gradient.  When used via
    ``get_activation``, the derivative entry is ``None``, and
    ``FeedForwardNetwork`` will skip the derivative multiplication during
    backpropagation accordingly.
    """
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def get_activation(name):
    """Look up an activation function and its derivative by name.

    Parameters
    ----------
    name : str
        One of ``'sigmoid'``, ``'tanh'``, ``'relu'``, ``'leaky_relu'``,
        or ``'softmax'``.

    Returns
    -------
    tuple of callable
        ``(forward_fn, derivative_fn)``.

    Raises
    ------
    ValueError
        If *name* is not a recognised activation function.
    """
    registry = {
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
