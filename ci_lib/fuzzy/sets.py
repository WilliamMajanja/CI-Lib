"""Fuzzy set definitions with common membership functions."""

from __future__ import annotations

from typing import Callable

import numpy as np


class FuzzySet:
    """A fuzzy set defined by a membership function.

    Parameters
    ----------
    name : str
        Human-readable label for the set.
    mf_func : callable
        Membership function mapping a crisp value to [0, 1].

    References
    ----------
    .. [1] Zadeh, L. A. (1965). Fuzzy sets. Information and Control, 8(3), 338-353.
    """

    def __init__(self, name: str, mf_func: Callable[[float], float]) -> None:
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name).__name__}")
        if not callable(mf_func):
            raise TypeError("mf_func must be callable")
        self.name = name
        self.mf_func = mf_func

    @classmethod
    def triangular(cls, name: str, a: float, b: float, c: float) -> FuzzySet:
        if not (a <= b <= c):
            raise ValueError(f"Require a <= b <= c, got a={a}, b={b}, c={c}")

        def _tri(x: float) -> float:
            if x <= a or x >= c:
                return 0.0
            if x <= b:
                return (x - a) / (b - a) if b != a else 1.0
            return (c - x) / (c - b) if c != b else 1.0

        return cls(name, _tri)

    @classmethod
    def trapezoidal(cls, name: str, a: float, b: float, c: float, d: float) -> FuzzySet:
        if not (a <= b <= c <= d):
            raise ValueError(f"Require a <= b <= c <= d, got a={a}, b={b}, c={c}, d={d}")

        def _trap(x: float) -> float:
            if x <= a or x >= d:
                return 0.0
            if x <= b:
                return (x - a) / (b - a) if b != a else 1.0
            if x <= c:
                return 1.0
            return (d - x) / (d - c) if d != c else 1.0

        return cls(name, _trap)

    @classmethod
    def gaussian(cls, name: str, mean: float, sigma: float) -> FuzzySet:
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        def _gauss(x: float) -> float:
            return float(np.exp(-0.5 * ((x - mean) / sigma) ** 2))

        return cls(name, _gauss)

    def degree(self, x: float) -> float:
        return float(np.clip(self.mf_func(x), 0.0, 1.0))

    def __and__(self, other: FuzzySet) -> FuzzySet:
        if not isinstance(other, FuzzySet):
            return NotImplemented
        name = f"({self.name} AND {other.name})"
        return FuzzySet(name, lambda x: min(self.degree(x), other.degree(x)))

    def __or__(self, other: FuzzySet) -> FuzzySet:
        if not isinstance(other, FuzzySet):
            return NotImplemented
        name = f"({self.name} OR {other.name})"
        return FuzzySet(name, lambda x: max(self.degree(x), other.degree(x)))

    def __invert__(self) -> FuzzySet:
        name = f"(NOT {self.name})"
        return FuzzySet(name, lambda x: 1.0 - self.degree(x))

    def __repr__(self) -> str:
        return f"FuzzySet(name={self.name!r})"
