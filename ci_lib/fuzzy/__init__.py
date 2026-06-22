"""Fuzzy logic module."""
from ci_lib.fuzzy.sets import FuzzySet
from ci_lib.fuzzy.variables import FuzzyVariable
from ci_lib.fuzzy.rules import FuzzyRule
from ci_lib.fuzzy.inference import FuzzyInferenceSystem

__all__ = ["FuzzySet", "FuzzyVariable", "FuzzyRule", "FuzzyInferenceSystem"]
