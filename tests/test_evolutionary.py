"""Tests for ci_lib.evolutionary (GeneticAlgorithm + DifferentialEvolution)."""

import numpy as np
import pytest

from ci_lib.evolutionary.genetic import GeneticAlgorithm
from ci_lib.evolutionary.differential import DifferentialEvolution


def sphere(x):
    return float(np.sum(x ** 2))


BOUNDS_3D = np.array([[-5.0, 5.0]] * 3)


# ------------------------------------------------------------------
# GeneticAlgorithm
# ------------------------------------------------------------------

class TestGeneticAlgorithm:
    def test_minimize_sphere(self):
        ga = GeneticAlgorithm(fitness_fn=sphere, n_dims=3, bounds=BOUNDS_3D,
                              pop_size=30, seed=42)
        best_sol, best_fit, history = ga.evolve(generations=80)
        assert best_fit < 1.0

    def test_bounds_enforcement(self):
        ga = GeneticAlgorithm(fitness_fn=sphere, n_dims=3, bounds=BOUNDS_3D,
                              pop_size=10, seed=42)
        best_sol, _, _ = ga.evolve(generations=20)
        assert np.all(best_sol >= -5.0) and np.all(best_sol <= 5.0)

    def test_history_non_increasing(self):
        ga = GeneticAlgorithm(fitness_fn=sphere, n_dims=3, bounds=BOUNDS_3D,
                              pop_size=20, seed=42)
        _, _, history = ga.evolve(generations=30)
        for i in range(1, len(history)):
            assert history[i] <= history[i - 1] + 1e-12

    def test_return_types(self):
        ga = GeneticAlgorithm(fitness_fn=sphere, n_dims=3, bounds=BOUNDS_3D,
                              pop_size=10, seed=42)
        best_sol, best_fit, history = ga.evolve(generations=5)
        assert isinstance(best_sol, np.ndarray)
        assert isinstance(best_fit, float)
        assert isinstance(history, list)
        assert best_sol.shape == (3,)


# ------------------------------------------------------------------
# DifferentialEvolution
# ------------------------------------------------------------------

class TestDifferentialEvolution:
    def test_minimize_sphere(self):
        de = DifferentialEvolution(fitness_fn=sphere, n_dims=3,
                                   bounds=BOUNDS_3D, pop_size=30, seed=42)
        best_sol, best_fit, history = de.evolve(generations=80)
        assert best_fit < 1.0

    @pytest.mark.parametrize("strategy", ["best/1/bin", "rand/1/bin"])
    def test_strategies(self, strategy):
        de = DifferentialEvolution(fitness_fn=sphere, n_dims=3,
                                   bounds=BOUNDS_3D, pop_size=20,
                                   strategy=strategy, seed=42)
        _, best_fit, _ = de.evolve(generations=50)
        assert best_fit < 5.0

    def test_return_types(self):
        de = DifferentialEvolution(fitness_fn=sphere, n_dims=3,
                                   bounds=BOUNDS_3D, pop_size=10, seed=42)
        best_sol, best_fit, history = de.evolve(generations=5)
        assert isinstance(best_sol, np.ndarray)
        assert isinstance(best_fit, float)
        assert isinstance(history, list)

    def test_history_length(self):
        gens = 25
        de = DifferentialEvolution(fitness_fn=sphere, n_dims=3,
                                   bounds=BOUNDS_3D, pop_size=10, seed=42)
        _, _, history = de.evolve(generations=gens)
        assert len(history) == gens
