"""Fuzzy linguistic variables."""

import numpy as np

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
        Number of discrete points used for universe sampling.
        Default is 100.

    Attributes
    ----------
    name : str
        Variable label.
    universe : numpy.ndarray
        Uniformly spaced sample of the universe of discourse.
    sets : dict
        Mapping of set names to :class:`FuzzySet` instances.

    Raises
    ------
    TypeError
        If *name* is not a string, *universe_range* is not a tuple/list
        of length 2, or *resolution* is not an integer.
    ValueError
        If the lower bound is not strictly less than the upper bound,
        or *resolution* is less than 2.

    Examples
    --------
    >>> temp = FuzzyVariable("temperature", (0, 100))
    >>> temp.add_set("hot", FuzzySet.triangular("hot", 60, 80, 100))
    >>> temp.fuzzify(75)
    {'hot': ...}
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, name, universe_range, resolution=100):
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name).__name__}")
        if not isinstance(universe_range, (tuple, list)) or len(universe_range) != 2:
            raise TypeError("universe_range must be a (low, high) tuple or list")
        if not isinstance(resolution, int):
            raise TypeError(
                f"resolution must be an int, got {type(resolution).__name__}"
            )

        low, high = float(universe_range[0]), float(universe_range[1])
        if low >= high:
            raise ValueError(
                f"Lower bound must be less than upper bound, got ({low}, {high})"
            )
        if resolution < 2:
            raise ValueError(f"resolution must be >= 2, got {resolution}")

        self.name = name
        self.universe = np.linspace(low, high, resolution)
        self.sets = {}

    # ------------------------------------------------------------------
    # Set management
    # ------------------------------------------------------------------

    def add_set(self, name, fuzzy_set):
        """Register a fuzzy set under this variable.

        Parameters
        ----------
        name : str
            Key used to reference the set (e.g. ``"low"``).
        fuzzy_set : FuzzySet
            The fuzzy set to register.

        Raises
        ------
        TypeError
            If *name* is not a string or *fuzzy_set* is not a
            :class:`FuzzySet`.
        """
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name).__name__}")
        if not isinstance(fuzzy_set, FuzzySet):
            raise TypeError(
                f"fuzzy_set must be a FuzzySet, got {type(fuzzy_set).__name__}"
            )
        self.sets[name] = fuzzy_set

    # ------------------------------------------------------------------
    # Fuzzification
    # ------------------------------------------------------------------

    def fuzzify(self, crisp_value):
        """Compute membership degrees for every registered set.

        Parameters
        ----------
        crisp_value : float
            A crisp numeric input.

        Returns
        -------
        dict
            Mapping ``{set_name: float}`` of membership degrees.
        """
        return {
            name: fs.degree(crisp_value) for name, fs in self.sets.items()
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self):
        set_names = list(self.sets.keys())
        return (
            f"FuzzyVariable(name={self.name!r}, "
            f"universe=[{self.universe[0]:.4g}, {self.universe[-1]:.4g}], "
            f"sets={set_names})"
        )
