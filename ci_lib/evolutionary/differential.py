"""Differential evolution optimiser."""

import numpy as np


class DifferentialEvolution:
    """Differential evolution for continuous minimisation.

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
    F : float, optional
        Differential weight (mutation scale factor).  Must be in (0, 2].
        Default is 0.8.
    CR : float, optional
        Crossover probability.  Must be in [0, 1].  Default is 0.9.
    strategy : str, optional
        Mutation/crossover strategy.  Supported values are
        ``'best/1/bin'`` and ``'rand/1/bin'``.  Default is
        ``'best/1/bin'``.
    seed : int or None, optional
        Random seed for reproducibility.  Default is None.

    Examples
    --------
    >>> import numpy as np
    >>> de = DifferentialEvolution(
    ...     fitness_fn=lambda x: float(np.sum(x ** 2)),
    ...     n_dims=3,
    ...     bounds=np.array([[-5, 5]] * 3),
    ...     seed=42,
    ... )
    >>> best_sol, best_fit, history = de.evolve(generations=50)
    """

    _STRATEGIES = {"best/1/bin", "rand/1/bin"}

    def __init__(
        self,
        fitness_fn,
        n_dims,
        bounds,
        pop_size=50,
        F=0.8,
        CR=0.9,
        strategy="best/1/bin",
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
        if not (0.0 < F <= 2.0):
            raise ValueError("F must be in (0, 2]")
        if not 0.0 <= CR <= 1.0:
            raise ValueError("CR must be in [0, 1]")
        if strategy not in self._STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Supported: {sorted(self._STRATEGIES)}"
            )

        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.strategy = strategy
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

    def _mutant(self, idx, population, best_idx):
        """Generate a mutant vector for the given target index.

        Parameters
        ----------
        idx : int
            Index of the current target vector.
        population : np.ndarray, shape (pop_size, n_dims)
            Current population.
        best_idx : int
            Index of the best individual in the population.

        Returns
        -------
        mutant : np.ndarray, shape (n_dims,)
            Mutant vector clipped to bounds.
        """
        indices = [i for i in range(self.pop_size) if i != idx]

        if self.strategy == "best/1/bin":
            r1, r2 = self._rng.choice(indices, size=2, replace=False)
            base = population[best_idx]
        else:  # rand/1/bin
            chosen = self._rng.choice(indices, size=3, replace=False)
            r1, r2 = chosen[1], chosen[2]
            base = population[chosen[0]]

        mutant = base + self.F * (population[r1] - population[r2])
        mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])
        return mutant

    def _crossover(self, target, mutant):
        """Binomial crossover between target and mutant vectors.

        Parameters
        ----------
        target : np.ndarray, shape (n_dims,)
            Current target vector.
        mutant : np.ndarray, shape (n_dims,)
            Mutant vector.

        Returns
        -------
        trial : np.ndarray, shape (n_dims,)
            Trial vector.
        """
        j_rand = self._rng.integers(self.n_dims)
        cross_mask = self._rng.random(self.n_dims) < self.CR
        cross_mask[j_rand] = True
        trial = np.where(cross_mask, mutant, target)
        return trial

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evolve(self, generations=100, verbose=False):
        """Run the differential evolution algorithm.

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
