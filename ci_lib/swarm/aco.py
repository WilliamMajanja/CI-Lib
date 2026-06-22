"""Ant colony optimiser for the travelling salesman problem."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class AntColonyOptimizer:
    """Ant colony optimiser for the symmetric travelling salesman problem.

    Parameters
    ----------
    distance_matrix : array_like, shape (n_cities, n_cities)
        Symmetric matrix of pairwise distances. Must be square, non-negative,
        with a zero diagonal.
    n_ants : int, optional
        Number of ants per iteration (default 20).
    alpha : float, optional
        Pheromone importance exponent (default 1.0).
    beta : float, optional
        Heuristic (inverse distance) importance exponent (default 2.0).
    evaporation : float, optional
        Pheromone evaporation rate in (0, 1] (default 0.5).
    q : float, optional
        Pheromone deposit constant, must be positive (default 100.0).
    seed : int or None, optional
        Random seed for reproducibility.

    References
    ----------
    .. [1] Dorigo, M., Maniezzo, V., & Colorni, A. (1996). Ant system:
           optimization by a colony of cooperating agents. IEEE Trans. SMC-B.
    .. [2] Dorigo, M., & Gambardella, L. M. (1997). Ant colony system: a
           cooperative learning approach to the traveling salesman problem.
           IEEE Trans. EC.
    """

    def __init__(
        self,
        distance_matrix: npt.ArrayLike,
        n_ants: int = 20,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation: float = 0.5,
        q: float = 100.0,
        seed: int | None = None,
    ) -> None:
        self.distances = np.asarray(distance_matrix, dtype=np.float64)

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
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def _build_route(
        self,
        pheromone: npt.NDArray[np.float64],
        heuristic: npt.NDArray[np.float64],
    ) -> list[int]:
        start = int(self._rng.integers(self.n_cities))
        visited = {start}
        route = [start]

        for _ in range(self.n_cities - 1):
            current = route[-1]
            unvisited = np.array([c for c in range(self.n_cities) if c not in visited])

            tau = pheromone[current, unvisited] ** self.alpha
            eta = heuristic[current, unvisited] ** self.beta
            probs = tau * eta
            total = probs.sum()

            if total == 0.0:
                probs = np.ones(len(unvisited))
            else:
                probs = probs / total

            next_city = int(self._rng.choice(unvisited, p=probs))
            route.append(next_city)
            visited.add(next_city)

        return route

    def _route_distance(self, route: list[int]) -> float:
        total = 0.0
        for i in range(len(route)):
            total += self.distances[route[i], route[(i + 1) % len(route)]]
        return float(total)

    def optimize(
        self, iterations: int = 100, verbose: bool = False
    ) -> tuple[list[int], float, list[float]]:
        pheromone = np.ones((self.n_cities, self.n_cities), dtype=np.float64)

        with np.errstate(divide="ignore", invalid="ignore"):
            heuristic = np.where(self.distances > 0, 1.0 / self.distances, 0.0)

        best_route: list[int] | None = None
        best_distance = float("inf")
        history: list[float] = []

        for it in range(iterations):
            routes: list[list[int]] = []
            distances: list[float] = []

            for _ in range(self.n_ants):
                route = self._build_route(pheromone, heuristic)
                dist = self._route_distance(route)
                routes.append(route)
                distances.append(dist)

            iter_best_idx = int(np.argmin(distances))
            iter_best_dist = float(distances[iter_best_idx])

            if iter_best_dist < best_distance:
                best_distance = iter_best_dist
                best_route = list(routes[iter_best_idx])

            history.append(best_distance)

            pheromone *= 1.0 - self.evaporation

            for route, dist in zip(routes, distances):
                deposit = self.q / dist if dist > 0.0 else 0.0
                for i in range(len(route)):
                    a, b = route[i], route[(i + 1) % len(route)]
                    pheromone[a, b] += deposit
                    pheromone[b, a] += deposit

            if verbose:
                print(f"Iteration {it + 1}/{iterations}  best={best_distance:.6e}")

        return best_route, float(best_distance), history
