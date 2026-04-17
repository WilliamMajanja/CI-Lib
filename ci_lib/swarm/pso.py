"""Particle swarm optimiser for continuous minimisation."""

import numpy as np


class ParticleSwarmOptimizer:
    """Particle swarm optimiser for continuous minimisation.

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
    n_particles : int, optional
        Number of particles in the swarm.  Must be >= 2.  Default is 30.
    w : float, optional
        Initial inertia weight.  Must be positive.  Default is 0.7.
    c1 : float, optional
        Cognitive acceleration coefficient.  Must be non-negative.
        Default is 1.5.
    c2 : float, optional
        Social acceleration coefficient.  Must be non-negative.
        Default is 1.5.
    w_decay : bool, optional
        If True, linearly decay inertia weight from ``w`` to 0.4 over
        the optimisation run.  Default is True.
    seed : int or None, optional
        Random seed for reproducibility.  Default is None.

    Examples
    --------
    >>> import numpy as np
    >>> pso = ParticleSwarmOptimizer(
    ...     fitness_fn=lambda x: float(np.sum(x ** 2)),
    ...     n_dims=3,
    ...     bounds=np.array([[-5, 5]] * 3),
    ...     seed=42,
    ... )
    >>> best_pos, best_fit, history = pso.optimize(iterations=50)
    """

    def __init__(
        self,
        fitness_fn,
        n_dims,
        bounds,
        n_particles=30,
        w=0.7,
        c1=1.5,
        c2=1.5,
        w_decay=True,
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

        if not isinstance(n_particles, int) or n_particles < 2:
            raise ValueError("n_particles must be an integer >= 2")
        if w <= 0.0:
            raise ValueError("w must be positive")
        if c1 < 0.0:
            raise ValueError("c1 must be non-negative")
        if c2 < 0.0:
            raise ValueError("c2 must be non-negative")

        self.n_particles = n_particles
        self.w = float(w)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.w_decay = w_decay
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize_swarm(self):
        """Create random particle positions and velocities within bounds.

        Returns
        -------
        positions : np.ndarray, shape (n_particles, n_dims)
            Initial positions.
        velocities : np.ndarray, shape (n_particles, n_dims)
            Initial velocities.
        """
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        positions = self._rng.uniform(
            lower, upper, size=(self.n_particles, self.n_dims)
        )
        span = upper - lower
        velocities = self._rng.uniform(
            -span, span, size=(self.n_particles, self.n_dims)
        )
        return positions, velocities

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self, iterations=100, verbose=False):
        """Run the particle swarm optimiser.

        Parameters
        ----------
        iterations : int, optional
            Number of iterations.  Default is 100.
        verbose : bool, optional
            If True, print progress every iteration.  Default is False.

        Returns
        -------
        best_position : np.ndarray, shape (n_dims,)
            Best position found.
        best_fitness : float
            Fitness of the best position.
        history : list of float
            Global best fitness value at each iteration.
        """
        positions, velocities = self._initialize_swarm()

        # Evaluate initial fitness
        fitnesses = np.array(
            [float(self.fitness_fn(pos)) for pos in positions]
        )

        # Personal bests
        personal_best_positions = positions.copy()
        personal_best_fitnesses = fitnesses.copy()

        # Global best
        g_best_idx = int(np.argmin(fitnesses))
        global_best_position = positions[g_best_idx].copy()
        global_best_fitness = float(fitnesses[g_best_idx])

        # Velocity clamp limits
        span = self.bounds[:, 1] - self.bounds[:, 0]
        v_max = span

        w_start = self.w
        w_end = 0.4
        history = []

        for it in range(iterations):
            # Compute current inertia weight
            if self.w_decay:
                w_current = w_start - (w_start - w_end) * it / max(iterations - 1, 1)
            else:
                w_current = w_start

            r1 = self._rng.random(size=(self.n_particles, self.n_dims))
            r2 = self._rng.random(size=(self.n_particles, self.n_dims))

            cognitive = self.c1 * r1 * (personal_best_positions - positions)
            social = self.c2 * r2 * (global_best_position - positions)
            velocities = w_current * velocities + cognitive + social

            # Velocity clamping
            velocities = np.clip(velocities, -v_max, v_max)

            # Update positions and clamp to bounds
            positions = positions + velocities
            positions = np.clip(positions, self.bounds[:, 0], self.bounds[:, 1])

            # Evaluate fitness
            fitnesses = np.array(
                [float(self.fitness_fn(pos)) for pos in positions]
            )

            # Update personal bests
            improved = fitnesses < personal_best_fitnesses
            personal_best_positions[improved] = positions[improved]
            personal_best_fitnesses[improved] = fitnesses[improved]

            # Update global best
            gen_best_idx = int(np.argmin(personal_best_fitnesses))
            gen_best_fit = float(personal_best_fitnesses[gen_best_idx])

            if gen_best_fit < global_best_fitness:
                global_best_fitness = gen_best_fit
                global_best_position = personal_best_positions[gen_best_idx].copy()

            history.append(global_best_fitness)

            if verbose:
                print(
                    f"Iteration {it + 1}/{iterations}  "
                    f"best={global_best_fitness:.6e}"
                )

        return global_best_position, float(global_best_fitness), history
