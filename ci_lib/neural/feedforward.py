"""Simple feed-forward neural network trained with back-propagation.

Uses only NumPy.  Supports mini-batch SGD, configurable activations,
and Xavier weight initialisation.
"""

import numpy as np

from ci_lib.neural.activations import get_activation


class FeedForwardNetwork:
    """A fully-connected feed-forward neural network.

    Parameters
    ----------
    layer_sizes : list of int
        Number of neurons in each layer, including input and output.
        Must contain at least two elements, each >= 1.
    activation : str, optional
        Name of the activation function (default ``'sigmoid'``).
    learning_rate : float, optional
        Step size for gradient descent (default ``0.01``).
    seed : int or None, optional
        Random seed for reproducibility.

    Attributes
    ----------
    weights : list of numpy.ndarray
        Weight matrices between consecutive layers.
    biases : list of numpy.ndarray
        Bias vectors for each layer after the input.

    Notes
    -----
    Activations whose derivative is ``None`` (e.g. softmax) are supported:
    the derivative multiplication is simply skipped during backpropagation,
    which is appropriate when the loss gradient is computed directly.

    Raises
    ------
    TypeError
        If *layer_sizes* is not a list of integers.
    ValueError
        If any layer size is less than 1 or the list has fewer than
        two elements.
    """

    def __init__(self, layer_sizes, activation="sigmoid", learning_rate=0.01,
                 seed=None):
        self._validate_layer_sizes(layer_sizes)
        self.layer_sizes = list(layer_sizes)
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)

        act_fn, act_deriv = get_activation(activation)
        self.activation = act_fn
        self.activation_derivative = act_deriv
        self.activation_name = activation

        # Xavier initialisation
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            w = self.rng.uniform(-limit, limit, size=(fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)

        # Populated during forward pass
        self._activations = []
        self._pre_activations = []

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_layer_sizes(layer_sizes):
        """Check that *layer_sizes* is a list of ints with len >= 2."""
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
                raise ValueError(
                    f"Each layer size must be >= 1, got {size}."
                )

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------
    def forward(self, X):
        """Compute a forward pass through the network.

        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            Network output of shape ``(n_samples, layer_sizes[-1])``.
        """
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

    def backward(self, X, y):
        """Run one backward pass and update weights.

        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape ``(n_samples, n_features)``.
        y : numpy.ndarray
            Target data of shape ``(n_samples, n_outputs)``.
        """
        X = np.atleast_2d(X)
        y = np.atleast_2d(y)
        m = X.shape[0]

        output = self.forward(X)

        # Output layer error
        error = output - y
        if self.activation_derivative is not None:
            delta = error * self.activation_derivative(self._pre_activations[-1])
        else:
            delta = error

        deltas = [delta]

        # Hidden layers (back to front)
        for i in range(len(self.weights) - 2, -1, -1):
            error = deltas[-1] @ self.weights[i + 1].T
            if self.activation_derivative is not None:
                delta = error * self.activation_derivative(self._pre_activations[i])
            else:
                delta = error
            deltas.append(delta)

        deltas.reverse()

        # Weight and bias updates
        for i in range(len(self.weights)):
            grad_w = self._activations[i].T @ deltas[i] / m
            grad_b = np.sum(deltas[i], axis=0, keepdims=True) / m
            self.weights[i] -= self.learning_rate * grad_w
            self.biases[i] -= self.learning_rate * grad_b

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X, y, epochs=1000, batch_size=None, verbose=False):
        """Train the network using (mini-batch) SGD.

        Parameters
        ----------
        X : numpy.ndarray
            Training inputs of shape ``(n_samples, n_features)``.
        y : numpy.ndarray
            Training targets of shape ``(n_samples, n_outputs)``.
        epochs : int, optional
            Number of passes over the dataset (default ``1000``).
        batch_size : int or None, optional
            Mini-batch size.  ``None`` means full-batch gradient descent.
        verbose : bool, optional
            If ``True``, print the MSE every 100 epochs.

        Returns
        -------
        FeedForwardNetwork
            The fitted network (``self``).
        """
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
                mse = self.score(X, y)
                print(f"Epoch {epoch:>5d} | MSE: {mse:.6f}")

        return self

    # ------------------------------------------------------------------
    # Prediction / evaluation
    # ------------------------------------------------------------------
    def predict(self, X):
        """Return the network output for *X*.

        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            Predictions of shape ``(n_samples, layer_sizes[-1])``.
        """
        return self.forward(X)

    def score(self, X, y):
        """Compute mean squared error on the given data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        y : numpy.ndarray
            Target data.

        Returns
        -------
        float
            Mean squared error (Python scalar).
        """
        y = np.atleast_2d(y)
        predictions = self.forward(X)
        return float(np.mean((predictions - y) ** 2))
