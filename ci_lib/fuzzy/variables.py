"""Fuzzy linguistic variables."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ci_lib.fuzzy.sets import FuzzySet


class FuzzyVariable:
    """A linguistic variable spanning a numeric universe of discourse.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. ``"temperature"``).
    universe_range : tuple of (float, float)
        ``(low, high)`` bounds of the universe of discourse.
    resolution : int, optional
        Number of discrete points for universe sampling (default 100).

    Attributes
    ----------
    name : str
        Variable label.
    universe : numpy.ndarray
        Uniformly spaced sample of the universe.
    sets : dict
        Mapping of set names to :class:`FuzzySet` instances.
    """

    def __init__(
        self,
        name: str,
        universe_range: tuple[float, float],
        resolution: int = 100,
    ) -> None:
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name).__name__}")
        if not isinstance(universe_range, (tuple, list)) or len(universe_range) != 2:
            raise TypeError("universe_range must be a (low, high) tuple or list")
        if not isinstance(resolution, int):
            raise TypeError(f"resolution must be an int, got {type(resolution).__name__}")

        low, high = float(universe_range[0]), float(universe_range[1])
        if low >= high:
            raise ValueError(f"Lower bound must be less than upper bound, got ({low}, {high})")
        if resolution < 2:
            raise ValueError(f"resolution must be >= 2, got {resolution}")

        self.name = name
        self.universe: npt.NDArray[np.float64] = np.linspace(low, high, resolution)
        self.sets: dict[str, FuzzySet] = {}

    def add_set(self, name: str, fuzzy_set: FuzzySet) -> None:
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name).__name__}")
        if not isinstance(fuzzy_set, FuzzySet):
            raise TypeError(f"fuzzy_set must be a FuzzySet, got {type(fuzzy_set).__name__}")
        self.sets[name] = fuzzy_set

    def fuzzify(self, crisp_value: float) -> dict[str, float]:
        return {name: fs.degree(crisp_value) for name, fs in self.sets.items()}

    def __repr__(self) -> str:
        set_names = list(self.sets.keys())
        return (
            f"FuzzyVariable(name={self.name!r}, "
            f"universe=[{self.universe[0]:.4g}, {self.universe[-1]:.4g}], "
            f"sets={set_names})"
        )
