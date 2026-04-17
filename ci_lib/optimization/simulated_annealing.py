"""Simulated annealing optimization algorithm."""

import numpy as np


class SimulatedAnnealing:
    """Simulated annealing optimizer for continuous objective functions.

    Parameters
    ----------
    objective_fn : callable
        Function to minimize. Accepts a 1-D array of length `n_dims` and
        returns a scalar.
    n_dims : int
        Number of dimensions of the search space.
    bounds : array_like, shape (n_dims, 2)
        Lower and upper bounds for each dimension.  Each row is
        ``[lower, upper]``.
    initial_temp : float, optional
        Starting temperature.  Default is ``100.0``.
    cooling_rate : float, optional
        Rate parameter for the cooling schedule.  For the exponential
        schedule this is the multiplicative factor applied each step
        (should be in (0, 1)).  Default is ``0.99``.
    min_temp : float, optional
        Temperature at which the algorithm stops.  Default is ``1e-8``.
    max_iter_per_temp : int, optional
        Number of candidate solutions evaluated at each temperature.
        Default is ``10``.
    cooling_schedule : {'exponential', 'linear', 'logarithmic'}, optional
        Strategy used to decrease temperature.  Default is ``'exponential'``.
    seed : int or None, optional
        Seed for the random number generator.  Default is ``None``.

    Attributes
    ----------
    best_solution : ndarray or None
        Best solution found after calling :meth:`optimize`.
    best_fitness : float or None
        Objective value at ``best_solution``.
    """

    _SCHEDULES = {"exponential", "linear", "logarithmic"}

    def __init__(
        self,
        objective_fn,
        n_dims,
        bounds,
        initial_temp=100.0,
        cooling_rate=0.99,
        min_temp=1e-8,
        max_iter_per_temp=10,
        cooling_schedule="exponential",
        seed=None,
    ):
        if cooling_schedule not in self._SCHEDULES:
            raise ValueError(
                f"Unknown cooling_schedule '{cooling_schedule}'. "
                f"Choose from {sorted(self._SCHEDULES)}."
            )

        self.objective_fn = objective_fn
        self.n_dims = int(n_dims)
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.initial_temp = float(initial_temp)
        self.cooling_rate = float(cooling_rate)
        self.min_temp = float(min_temp)
        self.max_iter_per_temp = int(max_iter_per_temp)
        self.cooling_schedule = cooling_schedule
        self._rng = np.random.default_rng(seed)

        if self.bounds.shape != (self.n_dims, 2):
            raise ValueError(
                f"bounds must have shape ({self.n_dims}, 2), "
                f"got {self.bounds.shape}."
            )

        self.best_solution = None
        self.best_fitness = None

    # ------------------------------------------------------------------
    # Cooling helpers
    # ------------------------------------------------------------------
    def _cool(self, temp, step):
        """Return the new temperature after one cooling step.

        Parameters
        ----------
        temp : float
            Current temperature.
        step : int
            Current outer-loop step index (0-based).

        Returns
        -------
        float
            Updated temperature.
        """
        if self.cooling_schedule == "exponential":
            return temp * self.cooling_rate
        if self.cooling_schedule == "linear":
            return self.initial_temp - step * self.cooling_rate
        # logarithmic
        return self.initial_temp / (1.0 + self.cooling_rate * np.log1p(step))

    # ------------------------------------------------------------------
    # Neighbour generation
    # ------------------------------------------------------------------
    def _neighbour(self, current, temp):
        """Generate a neighbour of *current* within bounds.

        Parameters
        ----------
        current : ndarray, shape (n_dims,)
            Current solution vector.
        temp : float
            Current temperature, used to scale the perturbation.

        Returns
        -------
        ndarray, shape (n_dims,)
            New candidate solution clipped to bounds.
        """
        scale = (self.bounds[:, 1] - self.bounds[:, 0]) * (temp / self.initial_temp)
        perturbation = self._rng.normal(0.0, scale)
        candidate = current + perturbation
        return np.clip(candidate, self.bounds[:, 0], self.bounds[:, 1])

    # ------------------------------------------------------------------
    # Main optimisation loop
    # ------------------------------------------------------------------
    def optimize(self, verbose=False):
        """Run the simulated annealing optimisation.

        Parameters
        ----------
        verbose : bool, optional
            If ``True``, print progress every temperature step.
            Default is ``False``.

        Returns
        -------
        best_solution : ndarray, shape (n_dims,)
            Best solution found.
        best_fitness : float
            Objective value at ``best_solution``.
        history : list of float
            Best objective value recorded after each temperature step.
        """
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        current = self._rng.uniform(lower, upper)
        current_fitness = float(self.objective_fn(current))

        best = current.copy()
        best_fitness = current_fitness

        temp = self.initial_temp
        history = [best_fitness]
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
                print(
                    f"Step {step:>5d} | Temp {temp:.6e} | "
                    f"Best fitness {best_fitness:.6e}"
                )

        self.best_solution = best
        self.best_fitness = float(best_fitness)
        return best, float(best_fitness), history
