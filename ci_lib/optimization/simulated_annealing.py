"""Simulated annealing optimisation for continuous objective functions."""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt


class SimulatedAnnealing:
    """Simulated annealing optimiser for continuous minimisation.

    Parameters
    ----------
    objective_fn : callable
        Function to minimise. Accepts a 1-D array and returns a scalar.
    n_dims : int
        Number of dimensions of the search space.
    bounds : array_like, shape (n_dims, 2)
        Lower and upper bounds for each dimension ``[lower, upper]``.
    initial_temp : float, optional
        Starting temperature (default 100.0).
    cooling_rate : float, optional
        Rate parameter for the cooling schedule (default 0.99).
    min_temp : float, optional
        Temperature threshold for stopping (default 1e-8).
    max_iter_per_temp : int, optional
        Candidate evaluations per temperature (default 10).
    cooling_schedule : {'exponential', 'linear', 'logarithmic'}, optional
        Strategy for decreasing temperature (default 'exponential').
    seed : int or None, optional
        Random seed for reproducibility.

    References
    ----------
    .. [1] Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983).
           Optimization by simulated annealing. Science.
    """

    _SCHEDULES = {"exponential", "linear", "logarithmic"}

    def __init__(
        self,
        objective_fn: Callable[[npt.NDArray[np.float64]], float],
        n_dims: int,
        bounds: npt.ArrayLike,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.99,
        min_temp: float = 1e-8,
        max_iter_per_temp: int = 10,
        cooling_schedule: str = "exponential",
        seed: int | None = None,
    ) -> None:
        if cooling_schedule not in self._SCHEDULES:
            raise ValueError(
                f"Unknown cooling_schedule '{cooling_schedule}'. "
                f"Choose from {sorted(self._SCHEDULES)}."
            )

        self.objective_fn = objective_fn
        self.n_dims = int(n_dims)
        self.bounds = np.asarray(bounds, dtype=np.float64)

        if self.bounds.shape != (self.n_dims, 2):
            raise ValueError(f"bounds must have shape ({self.n_dims}, 2), got {self.bounds.shape}.")

        self.initial_temp = float(initial_temp)
        self.cooling_rate = float(cooling_rate)
        self.min_temp = float(min_temp)
        self.max_iter_per_temp = int(max_iter_per_temp)
        self.cooling_schedule = cooling_schedule
        self._rng: np.random.Generator = np.random.default_rng(seed)

        self.best_solution: npt.NDArray[np.float64] | None = None
        self.best_fitness: float | None = None

    def _cool(self, temp: float, step: int) -> float:
        if self.cooling_schedule == "exponential":
            return temp * self.cooling_rate
        if self.cooling_schedule == "linear":
            return self.initial_temp - step * self.cooling_rate
        return self.initial_temp / (1.0 + self.cooling_rate * np.log1p(step))

    def _neighbour(
        self, current: npt.NDArray[np.float64], temp: float
    ) -> npt.NDArray[np.float64]:
        scale = (self.bounds[:, 1] - self.bounds[:, 0]) * (temp / self.initial_temp)
        perturbation = self._rng.normal(0.0, scale)
        candidate = current + perturbation
        return np.clip(candidate, self.bounds[:, 0], self.bounds[:, 1])

    def optimize(
        self, verbose: bool = False
    ) -> tuple[npt.NDArray[np.float64], float, list[float]]:
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        current = self._rng.uniform(lower, upper)
        current_fitness = float(self.objective_fn(current))

        best = current.copy()
        best_fitness = current_fitness

        temp = self.initial_temp
        history: list[float] = [best_fitness]
        step = 0

        while temp > self.min_temp:
            for _ in range(self.max_iter_per_temp):
                candidate = self._neighbour(current, temp)
                candidate_fitness = float(self.objective_fn(candidate))
                delta = candidate_fitness - current_fitness

                if delta < 0 or self._rng.random() < np.exp(-delta / temp):
                    current = candidate
                    current_fitness = candidate_fitness

                if current_fitness < best_fitness:
                    best = current.copy()
                    best_fitness = current_fitness

            step += 1
            temp = self._cool(temp, step)
            temp = max(temp, 0.0)
            history.append(best_fitness)

            if verbose:
                print(f"Step {step:>5d} | Temp {temp:.6e} | Best fitness {best_fitness:.6e}")

        self.best_solution = best
        self.best_fitness = float(best_fitness)
        return best, float(best_fitness), history
