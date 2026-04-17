"""Gradient descent optimization algorithms."""

import numpy as np


class GradientDescent:
    """Gradient-based optimizer supporting SGD, momentum, and Adam.

    Parameters
    ----------
    objective_fn : callable
        Scalar-valued function to minimise.  Accepts a 1-D array and returns
        a scalar.
    gradient_fn : callable or None, optional
        Function that returns the gradient of ``objective_fn``.  Accepts a
        1-D array and returns a 1-D array of the same length.  When ``None``,
        the gradient is approximated with central finite differences.
        Default is ``None``.
    n_dims : int or None, optional
        Number of dimensions.  Required only when ``gradient_fn`` is ``None``
        and ``x0`` is not supplied to :meth:`optimize`.  Default is ``None``.
    learning_rate : float, optional
        Step size.  Default is ``0.01``.
    momentum : float, optional
        Momentum coefficient (used by the ``'momentum'`` method).
        Default is ``0.9``.
    method : {'sgd', 'momentum', 'adam'}, optional
        Optimisation algorithm.  Default is ``'sgd'``.
    seed : int or None, optional
        Seed for the random number generator used to create a default
        starting point.  Default is ``None``.

    Attributes
    ----------
    best_solution : ndarray or None
        Best solution found after calling :meth:`optimize`.
    best_value : float or None
        Objective value at ``best_solution``.
    """

    _METHODS = {"sgd", "momentum", "adam"}

    def __init__(
        self,
        objective_fn,
        gradient_fn=None,
        n_dims=None,
        learning_rate=0.01,
        momentum=0.9,
        method="sgd",
        seed=None,
    ):
        if method not in self._METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Choose from {sorted(self._METHODS)}."
            )

        self.objective_fn = objective_fn
        self.gradient_fn = gradient_fn
        self.n_dims = None if n_dims is None else int(n_dims)
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.method = method
        self._rng = np.random.default_rng(seed)

        self.best_solution = None
        self.best_value = None

    # ------------------------------------------------------------------
    # Numerical gradient via central differences
    # ------------------------------------------------------------------
    def _numerical_gradient(self, x, eps=1e-7):
        """Approximate the gradient using central finite differences.

        Parameters
        ----------
        x : ndarray, shape (n,)
            Point at which to evaluate the gradient.
        eps : float, optional
            Perturbation size.  Default is ``1e-7``.

        Returns
        -------
        ndarray, shape (n,)
            Approximated gradient vector.
        """
        grad = np.empty_like(x)
        for i in range(x.shape[0]):
            x_fwd = x.copy()
            x_bwd = x.copy()
            x_fwd[i] += eps
            x_bwd[i] -= eps
            grad[i] = (self.objective_fn(x_fwd) - self.objective_fn(x_bwd)) / (
                2.0 * eps
            )
        return grad

    # ------------------------------------------------------------------
    # Gradient evaluation
    # ------------------------------------------------------------------
    def _grad(self, x):
        """Return the gradient at *x*.

        Parameters
        ----------
        x : ndarray, shape (n,)
            Point at which to evaluate the gradient.

        Returns
        -------
        ndarray, shape (n,)
            Gradient vector.
        """
        if self.gradient_fn is not None:
            return np.asarray(self.gradient_fn(x), dtype=np.float64)
        return self._numerical_gradient(x)

    # ------------------------------------------------------------------
    # Update rules
    # ------------------------------------------------------------------
    @staticmethod
    def _sgd_step(x, grad, lr, **_kw):
        """Plain stochastic gradient descent step.

        Parameters
        ----------
        x : ndarray
            Current solution.
        grad : ndarray
            Current gradient.
        lr : float
            Learning rate.

        Returns
        -------
        ndarray
            Updated solution.
        dict
            Empty state dict (SGD is stateless).
        """
        return x - lr * grad, {}

    @staticmethod
    def _momentum_step(x, grad, lr, *, velocity, mu, **_kw):
        """Gradient descent with momentum.

        Parameters
        ----------
        x : ndarray
            Current solution.
        grad : ndarray
            Current gradient.
        lr : float
            Learning rate.
        velocity : ndarray
            Accumulated velocity vector.
        mu : float
            Momentum coefficient.

        Returns
        -------
        ndarray
            Updated solution.
        dict
            Updated state containing the new velocity.
        """
        velocity = mu * velocity + grad
        x_new = x - lr * velocity
        return x_new, {"velocity": velocity, "mu": mu}

    @staticmethod
    def _adam_step(x, grad, lr, *, m, v, t, beta1=0.9, beta2=0.999, eps=1e-8, **_kw):
        """Adam optimiser step.

        Parameters
        ----------
        x : ndarray
            Current solution.
        grad : ndarray
            Current gradient.
        lr : float
            Learning rate.
        m : ndarray
            First moment estimate.
        v : ndarray
            Second moment estimate.
        t : int
            Current time step (1-based).
        beta1 : float, optional
            Exponential decay rate for first moment.
        beta2 : float, optional
            Exponential decay rate for second moment.
        eps : float, optional
            Small constant for numerical stability.

        Returns
        -------
        ndarray
            Updated solution.
        dict
            Updated state containing ``m``, ``v``, and ``t``.
        """
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad ** 2
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        x_new = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        return x_new, {"m": m, "v": v, "t": t + 1}

    # ------------------------------------------------------------------
    # Main optimisation loop
    # ------------------------------------------------------------------
    def optimize(self, x0=None, max_iter=1000, tol=1e-8, verbose=False):
        """Run the gradient descent optimisation.

        Parameters
        ----------
        x0 : array_like or None, optional
            Initial solution.  If ``None``, a random starting point is
            drawn from a standard normal distribution.  Default is ``None``.
        max_iter : int, optional
            Maximum number of iterations.  Default is ``1000``.
        tol : float, optional
            Convergence tolerance on the change in objective value between
            consecutive iterations.  Default is ``1e-8``.
        verbose : bool, optional
            If ``True``, print progress every 100 iterations.
            Default is ``False``.

        Returns
        -------
        best_solution : ndarray
            Best solution found.
        best_value : float
            Objective value at ``best_solution``.
        history : list of float
            Objective value recorded at each iteration.

        Raises
        ------
        ValueError
            If ``x0`` is ``None`` and ``n_dims`` was not provided at
            construction time.
        """
        if x0 is not None:
            x = np.asarray(x0, dtype=np.float64).copy()
        else:
            if self.n_dims is None:
                raise ValueError(
                    "x0 must be provided when n_dims was not set at init."
                )
            x = self._rng.standard_normal(self.n_dims)

        n = x.shape[0]

        # Initialise method-specific state
        if self.method == "momentum":
            state = {"velocity": np.zeros(n), "mu": self.momentum}
            step_fn = self._momentum_step
        elif self.method == "adam":
            state = {"m": np.zeros(n), "v": np.zeros(n), "t": 1}
            step_fn = self._adam_step
        else:
            state = {}
            step_fn = self._sgd_step

        best_x = x.copy()
        best_val = float(self.objective_fn(x))
        history = [best_val]

        for i in range(1, max_iter + 1):
            grad = self._grad(x)
            x, state = step_fn(x, grad, self.learning_rate, **state)

            val = float(self.objective_fn(x))
            history.append(val)

            if val < best_val:
                best_val = val
                best_x = x.copy()

            if verbose and i % 100 == 0:
                print(
                    f"Iter {i:>5d} | Value {val:.6e} | "
                    f"Best {best_val:.6e}"
                )

            if len(history) >= 2 and abs(history[-1] - history[-2]) < tol:
                break

        self.best_solution = best_x
        self.best_value = float(best_val)
        return best_x, float(best_val), history
