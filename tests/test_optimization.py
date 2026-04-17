"""Tests for ci_lib.optimization (SimulatedAnnealing + GradientDescent)."""

import numpy as np
import pytest

from ci_lib.optimization.simulated_annealing import SimulatedAnnealing
from ci_lib.optimization.gradient_descent import GradientDescent


def sphere(x):
    return float(np.sum(x ** 2))


BOUNDS_3D = np.array([[-5.0, 5.0]] * 3)


# ------------------------------------------------------------------
# SimulatedAnnealing
# ------------------------------------------------------------------

class TestSimulatedAnnealing:
    def test_minimize_sphere(self):
        sa = SimulatedAnnealing(objective_fn=sphere, n_dims=3, bounds=BOUNDS_3D,
                                initial_temp=100.0, cooling_rate=0.95, seed=42)
        best_sol, best_fit, history = sa.optimize()
        assert best_fit < 1.0

    @pytest.mark.parametrize("schedule,cooling_rate", [
        ("exponential", 0.5),
        ("linear", 5.0),
        ("logarithmic", 100.0),
    ])
    def test_cooling_schedules(self, schedule, cooling_rate):
        sa = SimulatedAnnealing(
            objective_fn=sphere, n_dims=3, bounds=BOUNDS_3D,
            initial_temp=10.0, cooling_rate=cooling_rate,
            min_temp=1.0, max_iter_per_temp=2,
            cooling_schedule=schedule, seed=42,
        )
        best_sol, best_fit, history = sa.optimize()
        assert isinstance(best_fit, float)

    def test_return_types(self):
        sa = SimulatedAnnealing(objective_fn=sphere, n_dims=3, bounds=BOUNDS_3D,
                                initial_temp=10.0, cooling_rate=0.9, seed=42)
        best_sol, best_fit, history = sa.optimize()
        assert isinstance(best_sol, np.ndarray)
        assert isinstance(best_fit, float)
        assert isinstance(history, list)


# ------------------------------------------------------------------
# GradientDescent
# ------------------------------------------------------------------

def x_squared(x):
    return float(np.sum(x ** 2))


def x_squared_grad(x):
    return 2.0 * x


class TestGradientDescent:
    def test_minimize_with_gradient(self):
        gd = GradientDescent(objective_fn=x_squared, gradient_fn=x_squared_grad,
                             learning_rate=0.1, method="sgd", seed=42)
        best_sol, best_val, history = gd.optimize(x0=np.array([3.0, -4.0]),
                                                   max_iter=200)
        assert best_val < 0.01

    def test_minimize_numerical_gradient(self):
        gd = GradientDescent(objective_fn=x_squared, n_dims=2,
                             learning_rate=0.1, method="sgd", seed=42)
        best_sol, best_val, history = gd.optimize(x0=np.array([3.0, -4.0]),
                                                   max_iter=200)
        assert best_val < 0.01

    @pytest.mark.parametrize("method", ["sgd", "momentum", "adam"])
    def test_methods(self, method):
        gd = GradientDescent(objective_fn=x_squared, gradient_fn=x_squared_grad,
                             learning_rate=0.05, method=method, seed=42)
        _, best_val, _ = gd.optimize(x0=np.array([2.0, -2.0]), max_iter=300)
        assert best_val < 1.0

    def test_history_decreases(self):
        gd = GradientDescent(objective_fn=x_squared, gradient_fn=x_squared_grad,
                             learning_rate=0.1, method="sgd", seed=42)
        _, _, history = gd.optimize(x0=np.array([5.0]), max_iter=50)
        assert history[-1] < history[0]

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            GradientDescent(objective_fn=x_squared, method="bogus")
