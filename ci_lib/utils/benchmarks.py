"""Classic benchmark optimisation functions for computational intelligence."""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt


class _BenchmarkFunction:
    """Base class for benchmark optimisation functions.

    Parameters
    ----------
    name : str
        Human-readable function name.
    optimum_value : float
        Known global minimum value.
    optimum_x : callable
        ``optimum_x(d)`` returns the global minimiser for dimension *d*.
    default_bounds : tuple of (float, float)
        ``(lower, upper)`` bounds applicable to every dimension.
    """

    def __init__(
        self,
        name: str,
        optimum_value: float,
        optimum_x: Callable[[int], npt.NDArray[np.float64]],
        default_bounds: tuple[float, float],
    ) -> None:
        self._name = name
        self._optimum_value = optimum_value
        self._optimum_x = optimum_x
        self._default_bounds = default_bounds

    @property
    def bounds(self) -> tuple[float, float]:
        return self._default_bounds

    def optimum(self, d: int = 2) -> npt.NDArray[np.float64]:
        return self._optimum_x(d)

    def __repr__(self) -> str:
        return f"{self._name}(optimum_value={self._optimum_value})"


class _Sphere(_BenchmarkFunction):
    r"""Sphere function: :math:`f(x) = \sum x_i^2`. Global minimum 0 at origin."""

    def __init__(self) -> None:
        super().__init__(
            name="Sphere",
            optimum_value=0.0,
            optimum_x=lambda d: np.zeros(d),
            default_bounds=(-5.12, 5.12),
        )

    def __call__(self, x: npt.ArrayLike) -> float:
        x = np.asarray(x, dtype=np.float64)
        return float(np.sum(x**2))


class _Rosenbrock(_BenchmarkFunction):
    r"""Rosenbrock function. Global minimum 0 at :math:`(1, ..., 1)`."""

    def __init__(self) -> None:
        super().__init__(
            name="Rosenbrock",
            optimum_value=0.0,
            optimum_x=lambda d: np.ones(d),
            default_bounds=(-5.0, 10.0),
        )

    def __call__(self, x: npt.ArrayLike) -> float:
        x = np.asarray(x, dtype=np.float64)
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


class _Rastrigin(_BenchmarkFunction):
    r"""Rastrigin function. Global minimum 0 at origin."""

    def __init__(self) -> None:
        super().__init__(
            name="Rastrigin",
            optimum_value=0.0,
            optimum_x=lambda d: np.zeros(d),
            default_bounds=(-5.12, 5.12),
        )

    def __call__(self, x: npt.ArrayLike) -> float:
        x = np.asarray(x, dtype=np.float64)
        d = len(x)
        return float(10.0 * d + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


class _Ackley(_BenchmarkFunction):
    r"""Ackley function. Global minimum 0 at origin."""

    def __init__(self) -> None:
        super().__init__(
            name="Ackley",
            optimum_value=0.0,
            optimum_x=lambda d: np.zeros(d),
            default_bounds=(-32.768, 32.768),
        )

    def __call__(self, x: npt.ArrayLike) -> float:
        x = np.asarray(x, dtype=np.float64)
        d = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(2.0 * np.pi * x))
        return float(
            -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / d))
            - np.exp(sum_cos / d) + 20.0 + np.e
        )


class _Griewank(_BenchmarkFunction):
    r"""Griewank function. Global minimum 0 at origin."""

    def __init__(self) -> None:
        super().__init__(
            name="Griewank",
            optimum_value=0.0,
            optimum_x=lambda d: np.zeros(d),
            default_bounds=(-600.0, 600.0),
        )

    def __call__(self, x: npt.ArrayLike) -> float:
        x = np.asarray(x, dtype=np.float64)
        indices = np.arange(1, len(x) + 1, dtype=np.float64)
        sum_part = np.sum(x**2) / 4000.0
        prod_part = np.prod(np.cos(x / np.sqrt(indices)))
        return float(1.0 + sum_part - prod_part)


class _Schwefel(_BenchmarkFunction):
    r"""Schwefel function. Global minimum 0 at :math:`(420.9687, ...)`."""

    _OPT = 420.9687

    def __init__(self) -> None:
        super().__init__(
            name="Schwefel",
            optimum_value=0.0,
            optimum_x=lambda d: np.full(d, _Schwefel._OPT),
            default_bounds=(-500.0, 500.0),
        )

    def __call__(self, x: npt.ArrayLike) -> float:
        x = np.asarray(x, dtype=np.float64)
        d = len(x)
        return float(418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


sphere = _Sphere()
rosenbrock = _Rosenbrock()
rastrigin = _Rastrigin()
ackley = _Ackley()
griewank = _Griewank()
schwefel = _Schwefel()
