"""Ant colony optimiser for the travelling salesman problem."""

import numpy as np


class AntColonyOptimizer:
    """Ant colony optimiser for the symmetric travelling salesman problem.

    Parameters
    ----------
    distance_matrix : array_like, shape (n_cities, n_cities)
        Symmetric matrix of pairwise distances.  Must be square with a
        non-negative, zero-diagonal structure.
    n_ants : int, optional
        Number of ants per iteration.  Must be >= 2.  Default is 20.
    alpha : float, optional
        Pheromone importance exponent.  Must be non-negative.
        Default is 1.0.
    beta : float, optional
        Heuristic (inverse distance) importance exponent.  Must be
        non-negative.  Default is 2.0.
    evaporation : float, optional
        Pheromone evaporation rate.  Must be in (0, 1].
        Default is 0.5.
    q : float, optional
        Pheromone deposit constant.  Must be positive.  Default is 100.0.
    seed : int or None, optional
        Random seed for reproducibility.  Default is None.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> coords = rng.random((6, 2))
    >>> d = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=-1))
    >>> aco = AntColonyOptimizer(d, seed=42)
    >>> best_route, best_dist, history = aco.optimize(iterations=30)
    """

    def __init__(
        self,
        distance_matrix,
        n_ants=20,
        alpha=1.0,
        beta=2.0,
        evaporation=0.5,
        q=100.0,
        seed=None,
    ):
        self.distances = np.asarray(distance_matrix, dtype=float)

        if self.distances.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")
        if self.distances.shape[0] != self.distances.shape[1]:
            raise ValueError("distance_matrix must be square")
        if self.distances.shape[0] < 2:
            raise ValueError("distance_matrix must have at least 2 cities")

        self.n_cities = self.distances.shape[0]

        if not isinstance(n_ants, int) or n_ants < 2:
            raise ValueError("n_ants must be an integer >= 2")
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if beta < 0.0:
            raise ValueError("beta must be non-negative")
        if not (0.0 < evaporation <= 1.0):
            raise ValueError("evaporation must be in (0, 1]")
        if q <= 0.0:
            raise ValueError("q must be positive")

        self.n_ants = n_ants
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.evaporation = float(evaporation)
        self.q = float(q)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_route(self, pheromone, heuristic):
        """Construct a single ant's tour using probabilistic selection.

        Parameters
        ----------
        pheromone : np.ndarray, shape (n_cities, n_cities)
            Current pheromone matrix.
        heuristic : np.ndarray, shape (n_cities, n_cities)
            Heuristic information (inverse distance).

        Returns
        -------
        route : list of int
            Ordered list of city indices forming a complete tour.
        """
        start = int(self._rng.integers(self.n_cities))
        visited = {start}
        route = [start]

        for _ in range(self.n_cities - 1):
            current = route[-1]
            unvisited = np.array(
                [c for c in range(self.n_cities) if c not in visited]
            )

            tau = pheromone[current, unvisited] ** self.alpha
            eta = heuristic[current, unvisited] ** self.beta
            probs = tau * eta
            total = probs.sum()

            if total == 0.0:
                # Fallback to uniform selection
                probs = np.ones(len(unvisited))
            else:
                probs = probs / total

            next_city = int(self._rng.choice(unvisited, p=probs))
            route.append(next_city)
            visited.add(next_city)

        return route

    def _route_distance(self, route):
        """Compute total tour distance for a given route.

        Parameters
        ----------
        route : list of int
            Ordered list of city indices.

        Returns
        -------
        distance : float
            Total round-trip distance.
        """
        total = 0.0
        for i in range(len(route)):
            total += self.distances[route[i], route[(i + 1) % len(route)]]
        return float(total)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self, iterations=100, verbose=False):
        """Run the ant colony optimiser.

        Parameters
        ----------
        iterations : int, optional
            Number of iterations.  Default is 100.
        verbose : bool, optional
            If True, print progress every iteration.  Default is False.

        Returns
        -------
        best_route : list of int
            Best tour found (list of city indices).
        best_distance : float
            Total distance of the best tour.
        history : list of float
            Best tour distance at each iteration.
        """
        # Initialise pheromone uniformly
        pheromone = np.ones((self.n_cities, self.n_cities))

        # Heuristic: inverse distance (guard against zero-distance)
        with np.errstate(divide="ignore", invalid="ignore"):
            heuristic = np.where(self.distances > 0, 1.0 / self.distances, 0.0)

        best_route = None
        best_distance = float("inf")
        history = []

        for it in range(iterations):
            routes = []
            distances = []

            for _ in range(self.n_ants):
                route = self._build_route(pheromone, heuristic)
                dist = self._route_distance(route)
                routes.append(route)
                distances.append(dist)

            # Update global best
            iter_best_idx = int(np.argmin(distances))
            iter_best_dist = float(distances[iter_best_idx])

            if iter_best_dist < best_distance:
                best_distance = iter_best_dist
                best_route = list(routes[iter_best_idx])

            history.append(best_distance)

            # Pheromone evaporation
            pheromone *= 1.0 - self.evaporation

            # Pheromone deposit
            for route, dist in zip(routes, distances):
                deposit = self.q / dist if dist > 0.0 else 0.0
                for i in range(len(route)):
                    a, b = route[i], route[(i + 1) % len(route)]
                    pheromone[a, b] += deposit
                    pheromone[b, a] += deposit

            if verbose:
                print(
                    f"Iteration {it + 1}/{iterations}  "
                    f"best={best_distance:.6e}"
                )

        return best_route, float(best_distance), history
