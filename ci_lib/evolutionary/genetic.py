"""Genetic algorithm with SBX crossover and Gaussian mutation."""

import numpy as np


class GeneticAlgorithm:
    """A genetic algorithm for continuous optimisation (minimisation).

    Parameters
    ----------
    fitness_fn : callable
        Objective function to minimise.  Accepts a 1-D array of length
        ``n_dims`` and returns a scalar.
    n_dims : int
        Number of decision variables.
    bounds : array_like, shape (n_dims, 2)
        Lower and upper bounds for each dimension.  Each row is
        ``[lower, upper]``.
    pop_size : int, optional
        Population size.  Must be >= 4.  Default is 50.
    mutation_rate : float, optional
        Probability of mutating each gene.  Must be in [0, 1].
        Default is 0.1.
    crossover_rate : float, optional
        Probability of performing crossover.  Must be in [0, 1].
        Default is 0.8.
    elitism : int, optional
        Number of elite individuals carried over unchanged each
        generation.  Must be >= 0 and < ``pop_size``.  Default is 2.
    seed : int or None, optional
        Random seed for reproducibility.  Default is None.

    Examples
    --------
    >>> import numpy as np
    >>> ga = GeneticAlgorithm(
    ...     fitness_fn=lambda x: float(np.sum(x ** 2)),
    ...     n_dims=3,
    ...     bounds=np.array([[-5, 5]] * 3),
    ...     seed=42,
    ... )
    >>> best_sol, best_fit, history = ga.evolve(generations=50)
    """

    def __init__(
        self,
        fitness_fn,
        n_dims,
        bounds,
        pop_size=50,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elitism=2,
        seed=None,
    ):
        if not callable(fitness_fn):
            raise TypeError("fitness_fn must be callable")
        if not isinstance(n_dims, int) or n_dims < 1:
            raise ValueError("n_dims must be a positive integer")

        self.fitness_fn = fitness_fn
        self.n_dims = n_dims
        self.bounds = np.asarray(bounds, dtype=float)

        if self.bounds.shape != (n_dims, 2):
            raise ValueError(
                f"bounds must have shape ({n_dims}, 2), "
                f"got {self.bounds.shape}"
            )
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError(
                "Each lower bound must be strictly less than the upper bound"
            )

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
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize_population(self):
        """Create a random population uniformly distributed within bounds.

        Returns
        -------
        population : np.ndarray, shape (pop_size, n_dims)
            Initial population.
        """
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        return self._rng.uniform(lower, upper, size=(self.pop_size, self.n_dims))

    def _tournament_select(self, fitnesses, k=3):
        """Select an individual via tournament selection.

        Parameters
        ----------
        fitnesses : np.ndarray, shape (pop_size,)
            Fitness values for the current population.
        k : int, optional
            Tournament size.  Default is 3.

        Returns
        -------
        index : int
            Index of the selected individual.
        """
        indices = self._rng.choice(self.pop_size, size=k, replace=False)
        winner = indices[np.argmin(fitnesses[indices])]
        return int(winner)

    def _crossover(self, p1, p2):
        """Simulated binary crossover (SBX) of two parents.

        Parameters
        ----------
        p1 : np.ndarray, shape (n_dims,)
            First parent.
        p2 : np.ndarray, shape (n_dims,)
            Second parent.

        Returns
        -------
        child : np.ndarray, shape (n_dims,)
            Offspring clipped to bounds.
        """
        if self._rng.random() > self.crossover_rate:
            return p1.copy()

        eta = 2.0  # distribution index
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

    def _mutate(self, individual):
        """Apply Gaussian mutation, clipping results to bounds.

        Parameters
        ----------
        individual : np.ndarray, shape (n_dims,)
            Individual to mutate.

        Returns
        -------
        mutated : np.ndarray, shape (n_dims,)
            Mutated individual.
        """
        mutated = individual.copy()
        for i in range(self.n_dims):
            if self._rng.random() < self.mutation_rate:
                span = self.bounds[i, 1] - self.bounds[i, 0]
                mutated[i] += self._rng.normal(0.0, 0.1 * span)
        mutated = np.clip(mutated, self.bounds[:, 0], self.bounds[:, 1])
        return mutated

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evolve(self, generations=100, verbose=False):
        """Run the genetic algorithm.

        Parameters
        ----------
        generations : int, optional
            Number of generations.  Default is 100.
        verbose : bool, optional
            If True, print progress every generation.  Default is False.

        Returns
        -------
        best_solution : np.ndarray, shape (n_dims,)
            Best solution found.
        best_fitness : float
            Fitness of the best solution.
        history : list of float
            Best fitness value at each generation.
        """
        population = self._initialize_population()
        fitnesses = np.array(
            [float(self.fitness_fn(ind)) for ind in population]
        )

        best_idx = int(np.argmin(fitnesses))
        best_solution = population[best_idx].copy()
        best_fitness = float(fitnesses[best_idx])
        history = []

        for gen in range(generations):
            # Sort by fitness for elitism
            order = np.argsort(fitnesses)
            new_population = [population[order[i]].copy() for i in range(self.elitism)]

            while len(new_population) < self.pop_size:
                i1 = self._tournament_select(fitnesses)
                i2 = self._tournament_select(fitnesses)
                child = self._crossover(population[i1], population[i2])
                child = self._mutate(child)
                new_population.append(child)

            population = np.array(new_population[: self.pop_size])
            fitnesses = np.array(
                [float(self.fitness_fn(ind)) for ind in population]
            )

            gen_best_idx = int(np.argmin(fitnesses))
            gen_best_fit = float(fitnesses[gen_best_idx])

            if gen_best_fit < best_fitness:
                best_fitness = gen_best_fit
                best_solution = population[gen_best_idx].copy()

            history.append(best_fitness)

            if verbose:
                print(f"Generation {gen + 1}/{generations}  best={best_fitness:.6e}")

        return best_solution, float(best_fitness), history
