"""Mamdani fuzzy inference system."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ci_lib.fuzzy.variables import FuzzyVariable
from ci_lib.fuzzy.rules import FuzzyRule


class FuzzyInferenceSystem:
    """A Mamdani-style fuzzy inference system.

    Fuzzifies crisp inputs, evaluates rules, clips output membership
    functions by firing strength, aggregates clipped sets (max), and
    defuzzifies to crisp outputs.

    Attributes
    ----------
    inputs : dict
        Registered input :class:`FuzzyVariable` instances keyed by name.
    outputs : dict
        Registered output :class:`FuzzyVariable` instances keyed by name.
    rules : list of FuzzyRule
        The rule base.

    References
    ----------
    .. [1] Mamdani, E. H., & Assilian, S. (1975). An experiment in linguistic
           synthesis with a fuzzy logic controller. Int. J. Man-Machine Studies.
    """

    _DEFUZZIFY_METHODS = {"centroid", "bisector", "max_membership"}

    def __init__(self) -> None:
        self.inputs: dict[str, FuzzyVariable] = {}
        self.outputs: dict[str, FuzzyVariable] = {}
        self.rules: list[FuzzyRule] = []

    def add_variable(self, variable: FuzzyVariable, var_type: str = "input") -> None:
        if not isinstance(variable, FuzzyVariable):
            raise TypeError(f"variable must be a FuzzyVariable, got {type(variable).__name__}")
        if var_type not in ("input", "output"):
            raise ValueError(f"var_type must be 'input' or 'output', got {var_type!r}")

        if var_type == "input":
            self.inputs[variable.name] = variable
        else:
            self.outputs[variable.name] = variable

    def add_rule(self, rule: FuzzyRule) -> None:
        if not isinstance(rule, FuzzyRule):
            raise TypeError(f"rule must be a FuzzyRule, got {type(rule).__name__}")
        self.rules.append(rule)

    def compute(
        self, inputs: dict[str, float], defuzzify_method: str = "centroid"
    ) -> dict[str, float]:
        if defuzzify_method not in self._DEFUZZIFY_METHODS:
            raise ValueError(
                f"Unknown defuzzify method {defuzzify_method!r}. "
                f"Choose from {sorted(self._DEFUZZIFY_METHODS)}."
            )

        rule_strengths = [(rule, rule.evaluate(inputs)) for rule in self.rules]

        results: dict[str, float] = {}
        for out_name, out_var in self.outputs.items():
            universe = out_var.universe
            aggregated = np.zeros_like(universe)

            for rule, strength in rule_strengths:
                con_var, con_set_name = rule.consequent
                if con_var.name != out_name:
                    continue
                mf = con_var.sets[con_set_name]
                mf_values = np.array([mf.degree(x) for x in universe])
                clipped = np.minimum(mf_values, strength)
                aggregated = np.maximum(aggregated, clipped)

            results[out_name] = self._defuzzify(universe, aggregated, defuzzify_method)

        return results

    def _defuzzify(
        self,
        universe: npt.NDArray[np.float64],
        aggregated: npt.NDArray[np.float64],
        method: str,
    ) -> float:
        if method == "centroid":
            return self._defuzzify_centroid(universe, aggregated)
        if method == "bisector":
            return self._defuzzify_bisector(universe, aggregated)
        return self._defuzzify_max_membership(universe, aggregated)

    @staticmethod
    def _defuzzify_centroid(
        universe: npt.NDArray[np.float64], aggregated: npt.NDArray[np.float64]
    ) -> float:
        total_area = np.sum(aggregated)
        if total_area == 0.0:
            raise ValueError(
                "Cannot defuzzify: aggregated membership area is zero "
                "(no rules fired or all firing strengths are zero)."
            )
        return float(np.sum(universe * aggregated) / total_area)

    @staticmethod
    def _defuzzify_bisector(
        universe: npt.NDArray[np.float64], aggregated: npt.NDArray[np.float64]
    ) -> float:
        total_area = np.sum(aggregated)
        if total_area == 0.0:
            raise ValueError(
                "Cannot defuzzify: aggregated membership area is zero "
                "(no rules fired or all firing strengths are zero)."
            )
        cumulative = np.cumsum(aggregated)
        half = total_area / 2.0
        idx = int(np.searchsorted(cumulative, half))
        idx = min(idx, len(universe) - 1)
        return float(universe[idx])

    @staticmethod
    def _defuzzify_max_membership(
        universe: npt.NDArray[np.float64], aggregated: npt.NDArray[np.float64]
    ) -> float:
        max_val = np.max(aggregated)
        if max_val == 0.0:
            raise ValueError(
                "Cannot defuzzify: aggregated membership area is zero "
                "(no rules fired or all firing strengths are zero)."
            )
        max_indices = np.where(aggregated == max_val)[0]
        return float(np.mean(universe[max_indices]))

    def __repr__(self) -> str:
        return (
            f"FuzzyInferenceSystem("
            f"inputs={list(self.inputs.keys())}, "
            f"outputs={list(self.outputs.keys())}, "
            f"rules={len(self.rules)})"
        )
