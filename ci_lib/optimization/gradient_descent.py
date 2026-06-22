"""Gradient descent optimisation with SGD, momentum, and Adam."""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt


class GradientDescent:
    """Gradient-based optimiser supporting SGD, momentum, and Adam.

    Parameters
    ----------
    objective_fn : callable
        Scalar-valued function to minimise.
    gradient_fn : callable or None, optional
        Function returning the gradient of *objective_fn*. If ``None``,
        the gradient is approximated via central finite differences.
    n_dims : int or None, optional
        Number of dimensions. Required when *gradient_fn* is ``None``
        and *x0* is not supplied to :meth:`optimize`.
    learning_rate : float, optional
        Step size (default 0.01).
    momentum : float, optional
        Momentum coefficient for the ``'momentum'`` method (default 0.9).
    method : {'sgd', 'momentum', 'adam'}, optional
        Optimisation algorithm (default 'sgd').
    seed : int or None, optional
        Random seed for reproducibility.

    References
    ----------
    .. [1] Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).
           Learning representations by back-propagating errors. Nature.
    .. [2] Kingma, D. P., & Ba, J. (2015). Adam: a method for stochastic
           optimization. ICLR.
    """

    _METHODS = {"sgd", "momentum", "adam"}

    def __init__(
        self,
        objective_fn: Callable[[npt.NDArray[np.float64]], float],
        gradient_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None = None,
        n_dims: int | None = None,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        method: str = "sgd",
        seed: int | None = None,
    ) -> None:
        if method not in self._METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from {sorted(self._METHODS)}.")

        self.objective_fn = objective_fn
        self.gradient_fn = gradient_fn
        self.n_dims = None if n_dims is None else int(n_dims)
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.method = method
        self._rng: np.random.Generator = np.random.default_rng(seed)

        self.best_solution: npt.NDArray[np.float64] | None = None
        self.best_value: float | None = None

    def _numerical_gradient(
        self, x: npt.NDArray[np.float64], eps: float = 1e-7
    ) -> npt.NDArray[np.float64]:
        grad = np.empty_like(x)
        for i in range(x.shape[0]):
            x_fwd = x.copy()
            x_bwd = x.copy()
            x_fwd[i] += eps
            x_bwd[i] -= eps
            grad[i] = (self.objective_fn(x_fwd) - self.objective_fn(x_bwd)) / (2.0 * eps)
        return grad

    def _grad(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.gradient_fn is not None:
            return np.asarray(self.gradient_fn(x), dtype=np.float64)
        return self._numerical_gradient(x)

    @staticmethod
    def _sgd_step(
        x: npt.NDArray[np.float64], grad: npt.NDArray[np.float64], lr: float, **_kw: object
    ) -> tuple[npt.NDArray[np.float64], dict[str, object]]:
        return x - lr * grad, {}

    @staticmethod
    def _momentum_step(
        x: npt.NDArray[np.float64],
        grad: npt.NDArray[np.float64],
        lr: float,
        *,
        velocity: npt.NDArray[np.float64],
        mu: float,
        **_kw: object,
    ) -> tuple[npt.NDArray[np.float64], dict[str, object]]:
        velocity = mu * velocity + grad
        x_new = x - lr * velocity
        return x_new, {"velocity": velocity, "mu": mu}

    @staticmethod
    def _adam_step(
        x: npt.NDArray[np.float64],
        grad: npt.NDArray[np.float64],
        lr: float,
        *,
        m: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
        t: int,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        **_kw: object,
    ) -> tuple[npt.NDArray[np.float64], dict[str, object]]:
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad**2
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        x_new = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        return x_new, {"m": m, "v": v, "t": t + 1}

    def optimize(
        self,
        x0: npt.ArrayLike | None = None,
        max_iter: int = 1000,
        tol: float = 1e-8,
        verbose: bool = False,
    ) -> tuple[npt.NDArray[np.float64], float, list[float]]:
        if x0 is not None:
            x = np.asarray(x0, dtype=np.float64).copy()
        else:
            if self.n_dims is None:
                raise ValueError("x0 must be provided when n_dims was not set at init.")
            x = self._rng.standard_normal(self.n_dims)

        n = x.shape[0]

        if self.method == "momentum":
            state: dict[str, object] = {"velocity": np.zeros(n), "mu": self.momentum}
            step_fn = self._momentum_step
        elif self.method == "adam":
            state = {"m": np.zeros(n), "v": np.zeros(n), "t": 1}
            step_fn = self._adam_step
        else:
            state = {}
            step_fn = self._sgd_step

        best_x = x.copy()
        best_val = float(self.objective_fn(x))
        history: list[float] = [best_val]

        for i in range(1, max_iter + 1):
            grad = self._grad(x)
            x, state = step_fn(x, grad, self.learning_rate, **state)

            val = float(self.objective_fn(x))
            history.append(val)

            if val < best_val:
                best_val = val
                best_x = x.copy()

            if verbose and i % 100 == 0:
                print(f"Iter {i:>5d} | Value {val:.6e} | Best {best_val:.6e}")

            if len(history) >= 2 and abs(history[-1] - history[-2]) < tol:
                break

        self.best_solution = best_x
        self.best_value = float(best_val)
        return best_x, float(best_val), history
