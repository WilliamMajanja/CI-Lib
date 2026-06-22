"""Genetic algorithm with SBX crossover and Gaussian mutation."""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt


class GeneticAlgorithm:
    """A genetic algorithm for continuous optimisation (minimisation).

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
    mutation_rate : float, optional
        Probability of mutating each gene in [0, 1] (default 0.1).
    crossover_rate : float, optional
        Probability of performing crossover in [0, 1] (default 0.8).
    elitism : int, optional
        Number of elite individuals carried over unchanged (default 2).
    seed : int or None, optional
        Random seed for reproducibility.

    References
    ----------
    .. [1] Holland, J. H. (1975). Adaptation in Natural and Artificial Systems.
    .. [2] Deb, K., & Agrawal, R. B. (1995). Simulated binary crossover for
           continuous search space. Complex Systems.
    """

    def __init__(
        self,
        fitness_fn: Callable[[npt.NDArray[np.float64]], float],
        n_dims: int,
        bounds: npt.ArrayLike,
        pop_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elitism: int = 2,
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
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        if not isinstance(elitism, int) or elitism < 0:
            raise ValueError("elitism must be a non-negative integer")
        if elitism >= pop_size:
            raise ValueError("elitism must be less than pop_size")

        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def _initialize_population(self) -> npt.NDArray[np.float64]:
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        return self._rng.uniform(lower, upper, size=(self.pop_size, self.n_dims))

    def _tournament_select(self, fitnesses: npt.NDArray[np.float64], k: int = 3) -> int:
        indices = self._rng.choice(self.pop_size, size=k, replace=False)
        winner = indices[np.argmin(fitnesses[indices])]
        return int(winner)

    def _crossover(
        self, p1: npt.NDArray[np.float64], p2: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        if self._rng.random() > self.crossover_rate:
            return p1.copy()

        eta = 2.0
        child = np.empty(self.n_dims)
        for i in range(self.n_dims):
            if abs(p1[i] - p2[i]) < 1e-14:
                child[i] = p1[i]
                continue
            u = self._rng.random()
            if u <= 0.5:
                beta = (2.0 * u) ** (1.0 / (eta + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
            child[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])

        child = np.clip(child, self.bounds[:, 0], self.bounds[:, 1])
        return child

    def _mutate(self, individual: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        mutated = individual.copy()
        for i in range(self.n_dims):
            if self._rng.random() < self.mutation_rate:
                span = self.bounds[i, 1] - self.bounds[i, 0]
                mutated[i] += self._rng.normal(0.0, 0.1 * span)
        mutated = np.clip(mutated, self.bounds[:, 0], self.bounds[:, 1])
        return mutated

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
            order = np.argsort(fitnesses)
            new_population = [population[order[i]].copy() for i in range(self.elitism)]

            while len(new_population) < self.pop_size:
                i1 = self._tournament_select(fitnesses)
                i2 = self._tournament_select(fitnesses)
                child = self._crossover(population[i1], population[i2])
                child = self._mutate(child)
                new_population.append(child)

            population = np.array(new_population[: self.pop_size])
            fitnesses = np.array([float(self.fitness_fn(ind)) for ind in population])

            gen_best_idx = int(np.argmin(fitnesses))
            gen_best_fit = float(fitnesses[gen_best_idx])

            if gen_best_fit < best_fitness:
                best_fitness = gen_best_fit
                best_solution = population[gen_best_idx].copy()

            history.append(best_fitness)

            if verbose:
                print(f"Generation {gen + 1}/{generations}  best={best_fitness:.6e}")

        return best_solution, float(best_fitness), history
