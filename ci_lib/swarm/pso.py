"""Particle swarm optimiser for continuous minimisation."""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt


class ParticleSwarmOptimizer:
    """Particle swarm optimiser for continuous minimisation.

    Parameters
    ----------
    fitness_fn : callable
        Objective function to minimise. Accepts a 1-D array and returns a scalar.
    n_dims : int
        Number of decision variables.
    bounds : array_like, shape (n_dims, 2)
        Lower and upper bounds for each dimension ``[lower, upper]``.
    n_particles : int, optional
        Number of particles in the swarm (default 30).
    w : float, optional
        Initial inertia weight, must be positive (default 0.7).
    c1 : float, optional
        Cognitive acceleration coefficient (default 1.5).
    c2 : float, optional
        Social acceleration coefficient (default 1.5).
    w_decay : bool, optional
        Linearly decay inertia weight from *w* to 0.4 (default True).
    seed : int or None, optional
        Random seed for reproducibility.

    References
    ----------
    .. [1] Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
           IEEE ICNN.
    .. [2] Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer.
           IEEE ICEC.
    """

    def __init__(
        self,
        fitness_fn: Callable[[npt.NDArray[np.float64]], float],
        n_dims: int,
        bounds: npt.ArrayLike,
        n_particles: int = 30,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        w_decay: bool = True,
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
            raise ValueError(f"bounds must have shape ({n_dims}, 2), got {self.bounds.shape}")
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("Each lower bound must be strictly less than the upper bound")

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
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def _initialize_swarm(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        positions = self._rng.uniform(lower, upper, size=(self.n_particles, self.n_dims))
        span = upper - lower
        velocities = self._rng.uniform(-span, span, size=(self.n_particles, self.n_dims))
        return positions, velocities

    def optimize(
        self, iterations: int = 100, verbose: bool = False
    ) -> tuple[npt.NDArray[np.float64], float, list[float]]:
        positions, velocities = self._initialize_swarm()

        fitnesses = np.array([float(self.fitness_fn(pos)) for pos in positions])

        personal_best_positions = positions.copy()
        personal_best_fitnesses = fitnesses.copy()

        g_best_idx = int(np.argmin(fitnesses))
        global_best_position = positions[g_best_idx].copy()
        global_best_fitness = float(fitnesses[g_best_idx])

        v_max = self.bounds[:, 1] - self.bounds[:, 0]

        w_start = self.w
        w_end = 0.4
        history: list[float] = []

        for it in range(iterations):
            if self.w_decay:
                w_current = w_start - (w_start - w_end) * it / max(iterations - 1, 1)
            else:
                w_current = w_start

            r1 = self._rng.random(size=(self.n_particles, self.n_dims))
            r2 = self._rng.random(size=(self.n_particles, self.n_dims))

            cognitive = self.c1 * r1 * (personal_best_positions - positions)
            social = self.c2 * r2 * (global_best_position - positions)
            velocities = w_current * velocities + cognitive + social
            velocities = np.clip(velocities, -v_max, v_max)

            positions = positions + velocities
            positions = np.clip(positions, self.bounds[:, 0], self.bounds[:, 1])

            fitnesses = np.array([float(self.fitness_fn(pos)) for pos in positions])

            improved = fitnesses < personal_best_fitnesses
            personal_best_positions[improved] = positions[improved]
            personal_best_fitnesses[improved] = fitnesses[improved]

            gen_best_idx = int(np.argmin(personal_best_fitnesses))
            gen_best_fit = float(personal_best_fitnesses[gen_best_idx])

            if gen_best_fit < global_best_fitness:
                global_best_fitness = gen_best_fit
                global_best_position = personal_best_positions[gen_best_idx].copy()

            history.append(global_best_fitness)

            if verbose:
                print(f"Iteration {it + 1}/{iterations}  best={global_best_fitness:.6e}")

        return global_best_position, float(global_best_fitness), history
