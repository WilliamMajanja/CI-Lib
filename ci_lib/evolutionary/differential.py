"""Differential evolution optimiser for continuous minimisation."""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt


class DifferentialEvolution:
    """Differential evolution for continuous minimisation.

    Parameters
    ----------
    fitness_fn : callable
        Objective function to minimise. Accepts a 1-D array and returns a scalar.
    n_dims : int
        Number of decision variables.
    bounds : array_like, shape (n_dims, 2)
        Lower and upper bounds for each dimension ``[lower, upper]``.
    pop_size : int, optional
        Population size, must be >= 4 (default 50).
    F : float, optional
        Differential weight (mutation scale) in (0, 2] (default 0.8).
    CR : float, optional
        Crossover probability in [0, 1] (default 0.9).
    strategy : {'best/1/bin', 'rand/1/bin'}, optional
        Mutation/crossover strategy (default 'best/1/bin').
    seed : int or None, optional
        Random seed for reproducibility.

    References
    ----------
    .. [1] Storn, R., & Price, K. (1997). Differential evolution — a simple
           and efficient heuristic for global optimization. J. Global Optimization.
    """

    _STRATEGIES = {"best/1/bin", "rand/1/bin"}

    def __init__(
        self,
        fitness_fn: Callable[[npt.NDArray[np.float64]], float],
        n_dims: int,
        bounds: npt.ArrayLike,
        pop_size: int = 50,
        F: float = 0.8,
        CR: float = 0.9,
        strategy: str = "best/1/bin",
        seed: int | None = None,
    ) -> None:
        if not callable(fitness_fn):
            raise TypeError("fitness_fn must be callable")
        if not isinstance(n_dims, int) or n_dims < 1:
            raise ValueError("n_dims must be a positive integer")

        self.fitness_fn = fitness_fn
        self.n_dims = n_dims
        self.bounds = np.asarray(bounds, dtype=np.float64)

        if self.bounds.shape != (n_dims, 2):
            raise ValueError(
                f"bounds must have shape ({n_dims}, 2), got {self.bounds.shape}"
            )
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("Each lower bound must be strictly less than the upper bound")

        if not isinstance(pop_size, int) or pop_size < 4:
            raise ValueError("pop_size must be an integer >= 4")
        if not (0.0 < F <= 2.0):
            raise ValueError("F must be in (0, 2]")
        if not 0.0 <= CR <= 1.0:
            raise ValueError("CR must be in [0, 1]")
        if strategy not in self._STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Supported: {sorted(self._STRATEGIES)}"
            )

        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.strategy = strategy
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def _initialize_population(self) -> npt.NDArray[np.float64]:
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        return self._rng.uniform(lower, upper, size=(self.pop_size, self.n_dims))

    def _mutant(
        self,
        idx: int,
        population: npt.NDArray[np.float64],
        best_idx: int,
    ) -> npt.NDArray[np.float64]:
        indices = [i for i in range(self.pop_size) if i != idx]

        if self.strategy == "best/1/bin":
            r1, r2 = self._rng.choice(indices, size=2, replace=False)
            base = population[best_idx]
        else:
            chosen = self._rng.choice(indices, size=3, replace=False)
            r1, r2 = chosen[1], chosen[2]
            base = population[chosen[0]]

        mutant = base + self.F * (population[r1] - population[r2])
        mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])
        return mutant

    def _crossover(
        self, target: npt.NDArray[np.float64], mutant: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        j_rand = self._rng.integers(self.n_dims)
        cross_mask = self._rng.random(self.n_dims) < self.CR
        cross_mask[j_rand] = True
        trial = np.where(cross_mask, mutant, target)
        return trial

    def evolve(
        self, generations: int = 100, verbose: bool = False
    ) -> tuple[npt.NDArray[np.float64], float, list[float]]:
        population = self._initialize_population()
        fitnesses = np.array([float(self.fitness_fn(ind)) for ind in population])

        best_idx = int(np.argmin(fitnesses))
        best_solution = population[best_idx].copy()
        best_fitness = float(fitnesses[best_idx])
        history: list[float] = []

        for gen in range(generations):
            for i in range(self.pop_size):
                mutant = self._mutant(i, population, best_idx)
                trial = self._crossover(population[i], mutant)
                trial_fitness = float(self.fitness_fn(trial))

                if trial_fitness <= fitnesses[i]:
                    population[i] = trial
                    fitnesses[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial.copy()
                        best_idx = i

            history.append(best_fitness)

            if verbose:
                print(f"Generation {gen + 1}/{generations}  best={best_fitness:.6e}")

        return best_solution, float(best_fitness), history
