"""Fuzzy set definitions with common membership functions."""

import numpy as np


class FuzzySet:
    """A fuzzy set defined by a membership function.

    Parameters
    ----------
    name : str
        Human-readable label for the set.
    mf_func : callable
        Membership function mapping a crisp value to [0, 1].
        Signature: ``mf_func(x) -> float``.

    Attributes
    ----------
    name : str
        Label of the fuzzy set.
    mf_func : callable
        The membership function.

    Raises
    ------
    TypeError
        If *name* is not a string or *mf_func* is not callable.

    Examples
    --------
    >>> fs = FuzzySet.triangular("medium", 2, 5, 8)
    >>> fs.degree(5)
    1.0
    >>> fs.degree(0)
    0.0
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, name, mf_func):
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name).__name__}")
        if not callable(mf_func):
            raise TypeError("mf_func must be callable")
        self.name = name
        self.mf_func = mf_func

    # ------------------------------------------------------------------
    # Class-method constructors for common shapes
    # ------------------------------------------------------------------

    @classmethod
    def triangular(cls, name, a, b, c):
        """Create a triangular membership function.

        Parameters
        ----------
        name : str
            Label for the fuzzy set.
        a : float
            Left foot (membership = 0).
        b : float
            Peak (membership = 1).
        c : float
            Right foot (membership = 0).

        Returns
        -------
        FuzzySet
            A new fuzzy set with a triangular membership function.

        Raises
        ------
        ValueError
            If *a*, *b*, *c* are not in non-decreasing order.
        """
        if not (a <= b <= c):
            raise ValueError(f"Require a <= b <= c, got a={a}, b={b}, c={c}")

        def _tri(x):
            x = float(x)
            if x <= a or x >= c:
                return 0.0
            if x <= b:
                return (x - a) / (b - a) if b != a else 1.0
            return (c - x) / (c - b) if c != b else 1.0

        return cls(name, _tri)

    @classmethod
    def trapezoidal(cls, name, a, b, c, d):
        """Create a trapezoidal membership function.

        Parameters
        ----------
        name : str
            Label for the fuzzy set.
        a : float
            Left foot (membership begins rising).
        b : float
            Left shoulder (membership reaches 1).
        c : float
            Right shoulder (membership begins falling).
        d : float
            Right foot (membership returns to 0).

        Returns
        -------
        FuzzySet
            A new fuzzy set with a trapezoidal membership function.

        Raises
        ------
        ValueError
            If *a*, *b*, *c*, *d* are not in non-decreasing order.
        """
        if not (a <= b <= c <= d):
            raise ValueError(
                f"Require a <= b <= c <= d, got a={a}, b={b}, c={c}, d={d}"
            )

        def _trap(x):
            x = float(x)
            if x <= a or x >= d:
                return 0.0
            if x <= b:
                return (x - a) / (b - a) if b != a else 1.0
            if x <= c:
                return 1.0
            return (d - x) / (d - c) if d != c else 1.0

        return cls(name, _trap)

    @classmethod
    def gaussian(cls, name, mean, sigma):
        """Create a Gaussian membership function.

        Parameters
        ----------
        name : str
            Label for the fuzzy set.
        mean : float
            Centre of the bell curve.
        sigma : float
            Standard deviation (width) of the bell curve.

        Returns
        -------
        FuzzySet
            A new fuzzy set with a Gaussian membership function.

        Raises
        ------
        ValueError
            If *sigma* is not positive.
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        def _gauss(x):
            return float(np.exp(-0.5 * ((float(x) - mean) / sigma) ** 2))

        return cls(name, _gauss)

    # ------------------------------------------------------------------
    # Membership evaluation
    # ------------------------------------------------------------------

    def degree(self, x):
        """Return the membership degree for crisp value *x*.

        Parameters
        ----------
        x : float
            Crisp input value.

        Returns
        -------
        float
            Membership degree clipped to [0, 1].
        """
        return float(np.clip(self.mf_func(x), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Fuzzy operators (return new FuzzySet instances)
    # ------------------------------------------------------------------

    def __and__(self, other):
        """Fuzzy intersection (min).

        Parameters
        ----------
        other : FuzzySet
            Another fuzzy set.

        Returns
        -------
        FuzzySet
            A new set whose membership is the minimum of both sets.
        """
        if not isinstance(other, FuzzySet):
            return NotImplemented
        name = f"({self.name} AND {other.name})"
        return FuzzySet(name, lambda x: min(self.degree(x), other.degree(x)))

    def __or__(self, other):
        """Fuzzy union (max).

        Parameters
        ----------
        other : FuzzySet
            Another fuzzy set.

        Returns
        -------
        FuzzySet
            A new set whose membership is the maximum of both sets.
        """
        if not isinstance(other, FuzzySet):
            return NotImplemented
        name = f"({self.name} OR {other.name})"
        return FuzzySet(name, lambda x: max(self.degree(x), other.degree(x)))

    def __invert__(self):
        """Fuzzy complement (1 - degree).

        Returns
        -------
        FuzzySet
            A new set whose membership is 1 minus the original.
        """
        name = f"(NOT {self.name})"
        return FuzzySet(name, lambda x: 1.0 - self.degree(x))

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self):
        return f"FuzzySet(name={self.name!r})"
