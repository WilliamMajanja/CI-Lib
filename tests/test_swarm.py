"""Tests for ci_lib.swarm (PSO + ACO)."""

import numpy as np
import pytest

from ci_lib.swarm.pso import ParticleSwarmOptimizer
from ci_lib.swarm.aco import AntColonyOptimizer


def sphere(x):
    return float(np.sum(x ** 2))


BOUNDS_3D = np.array([[-5.0, 5.0]] * 3)


# ------------------------------------------------------------------
# ParticleSwarmOptimizer
# ------------------------------------------------------------------

class TestPSO:
    def test_minimize_sphere(self):
        pso = ParticleSwarmOptimizer(fitness_fn=sphere, n_dims=3,
                                     bounds=BOUNDS_3D, n_particles=20, seed=42)
        best_pos, best_fit, history = pso.optimize(iterations=80)
        assert best_fit < 1.0

    def test_bounds_enforcement(self):
        pso = ParticleSwarmOptimizer(fitness_fn=sphere, n_dims=3,
                                     bounds=BOUNDS_3D, n_particles=10, seed=42)
        best_pos, _, _ = pso.optimize(iterations=30)
        assert np.all(best_pos >= -5.0) and np.all(best_pos <= 5.0)

    def test_history_length(self):
        iters = 40
        pso = ParticleSwarmOptimizer(fitness_fn=sphere, n_dims=3,
                                     bounds=BOUNDS_3D, n_particles=10, seed=42)
        _, _, history = pso.optimize(iterations=iters)
        assert len(history) == iters

    def test_return_types(self):
        pso = ParticleSwarmOptimizer(fitness_fn=sphere, n_dims=3,
                                     bounds=BOUNDS_3D, n_particles=5, seed=42)
        best_pos, best_fit, history = pso.optimize(iterations=5)
        assert isinstance(best_pos, np.ndarray)
        assert isinstance(best_fit, float)
        assert best_pos.shape == (3,)


# ------------------------------------------------------------------
# AntColonyOptimizer
# ------------------------------------------------------------------

class TestACO:
    @pytest.fixture
    def distance_matrix(self):
        return np.array([
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0],
        ], dtype=float)

    def test_valid_route(self, distance_matrix):
        aco = AntColonyOptimizer(distance_matrix, n_ants=5, seed=42)
        best_route, best_dist, history = aco.optimize(iterations=20)
        assert sorted(best_route) == [0, 1, 2, 3]

    def test_distance_positive(self, distance_matrix):
        aco = AntColonyOptimizer(distance_matrix, n_ants=5, seed=42)
        _, best_dist, _ = aco.optimize(iterations=10)
        assert best_dist > 0.0

    def test_history_length(self, distance_matrix):
        iters = 15
        aco = AntColonyOptimizer(distance_matrix, n_ants=5, seed=42)
        _, _, history = aco.optimize(iterations=iters)
        assert len(history) == iters
