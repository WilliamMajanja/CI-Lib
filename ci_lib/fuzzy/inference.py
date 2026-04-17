"""Mamdani fuzzy inference system."""

import numpy as np

from ci_lib.fuzzy.variables import FuzzyVariable
from ci_lib.fuzzy.rules import FuzzyRule


class FuzzyInferenceSystem:
    """A Mamdani-style fuzzy inference system.

    The system fuzzifies crisp inputs, evaluates rules, clips output
    membership functions by each rule's firing strength, aggregates
    the clipped sets (max), and defuzzifies to crisp outputs.

    Attributes
    ----------
    inputs : dict
        Registered input :class:`FuzzyVariable` instances keyed by name.
    outputs : dict
        Registered output :class:`FuzzyVariable` instances keyed by name.
    rules : list of FuzzyRule
        The rule base.

    Examples
    --------
    >>> fis = FuzzyInferenceSystem()
    >>> fis.add_variable(temp, var_type="input")
    >>> fis.add_variable(fan, var_type="output")
    >>> fis.add_rule(rule)
    >>> fis.compute({"temperature": 75})
    {'fan_speed': 62.5}
    """

    _DEFUZZIFY_METHODS = {"centroid", "bisector", "max_membership"}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self):
        self.inputs = {}
        self.outputs = {}
        self.rules = []

    # ------------------------------------------------------------------
    # Variable / rule registration
    # ------------------------------------------------------------------

    def add_variable(self, variable, var_type="input"):
        """Register an input or output variable.

        Parameters
        ----------
        variable : FuzzyVariable
            The variable to register.
        var_type : {'input', 'output'}
            Whether the variable is an input or output.

        Raises
        ------
        TypeError
            If *variable* is not a :class:`FuzzyVariable`.
        ValueError
            If *var_type* is not ``'input'`` or ``'output'``.
        """
        if not isinstance(variable, FuzzyVariable):
            raise TypeError(
                f"variable must be a FuzzyVariable, got {type(variable).__name__}"
            )
        if var_type not in ("input", "output"):
            raise ValueError(f"var_type must be 'input' or 'output', got {var_type!r}")

        if var_type == "input":
            self.inputs[variable.name] = variable
        else:
            self.outputs[variable.name] = variable

    def add_rule(self, rule):
        """Append a rule to the rule base.

        Parameters
        ----------
        rule : FuzzyRule
            The rule to add.

        Raises
        ------
        TypeError
            If *rule* is not a :class:`FuzzyRule`.
        """
        if not isinstance(rule, FuzzyRule):
            raise TypeError(
                f"rule must be a FuzzyRule, got {type(rule).__name__}"
            )
        self.rules.append(rule)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def compute(self, inputs, defuzzify_method="centroid"):
        """Run the full Mamdani inference pipeline.

        Parameters
        ----------
        inputs : dict
            Mapping ``{variable_name: crisp_value}`` for every
            registered input variable.
        defuzzify_method : {'centroid', 'bisector', 'max_membership'}
            Defuzzification strategy.  Default is ``'centroid'``.

        Returns
        -------
        dict
            Mapping ``{output_name: float}`` of defuzzified crisp
            values for each output variable.

        Raises
        ------
        ValueError
            If *defuzzify_method* is unknown, or no rules fire for an
            output, or aggregation yields a zero-area output.
        KeyError
            If *inputs* is missing a required input variable.
        """
        if defuzzify_method not in self._DEFUZZIFY_METHODS:
            raise ValueError(
                f"Unknown defuzzify method {defuzzify_method!r}. "
                f"Choose from {sorted(self._DEFUZZIFY_METHODS)}."
            )

        # 1. Evaluate every rule to get firing strengths
        rule_strengths = []
        for rule in self.rules:
            rule_strengths.append((rule, rule.evaluate(inputs)))

        # 2. For each output variable, aggregate clipped consequent MFs
        results = {}
        for out_name, out_var in self.outputs.items():
            universe = out_var.universe
            aggregated = np.zeros_like(universe)

            for rule, strength in rule_strengths:
                con_var, con_set_name = rule.consequent
                if con_var.name != out_name:
                    continue
                mf = con_var.sets[con_set_name]
                # Vectorised evaluation + clipping by firing strength
                mf_values = np.array([mf.degree(x) for x in universe])
                clipped = np.minimum(mf_values, strength)
                aggregated = np.maximum(aggregated, clipped)

            results[out_name] = self._defuzzify(
                universe, aggregated, defuzzify_method
            )

        return results

    # ------------------------------------------------------------------
    # Defuzzification
    # ------------------------------------------------------------------

    def _defuzzify(self, universe, aggregated, method):
        """Dispatch to the chosen defuzzification method.

        Parameters
        ----------
        universe : numpy.ndarray
            Discrete universe of discourse.
        aggregated : numpy.ndarray
            Aggregated membership values over *universe*.
        method : str
            Defuzzification strategy name.

        Returns
        -------
        float
            Crisp output value.
        """
        if method == "centroid":
            return self._defuzzify_centroid(universe, aggregated)
        if method == "bisector":
            return self._defuzzify_bisector(universe, aggregated)
        return self._defuzzify_max_membership(universe, aggregated)

    # ------------------------------------------------------------------
    # Centroid (centre of gravity)
    # ------------------------------------------------------------------

    @staticmethod
    def _defuzzify_centroid(universe, aggregated):
        """Centroid defuzzification.

        Parameters
        ----------
        universe : numpy.ndarray
            Discrete universe points.
        aggregated : numpy.ndarray
            Aggregated membership values.

        Returns
        -------
        float
            Centroid crisp value.

        Raises
        ------
        ValueError
            If the total area is zero.
        """
        total_area = np.sum(aggregated)
        if total_area == 0.0:
            raise ValueError(
                "Cannot defuzzify: aggregated membership area is zero "
                "(no rules fired or all firing strengths are zero)."
            )
        return float(np.sum(universe * aggregated) / total_area)

    # ------------------------------------------------------------------
    # Bisector
    # ------------------------------------------------------------------

    @staticmethod
    def _defuzzify_bisector(universe, aggregated):
        """Bisector defuzzification.

        The bisector is the universe value that splits the aggregated
        area into two equal halves.

        Parameters
        ----------
        universe : numpy.ndarray
            Discrete universe points.
        aggregated : numpy.ndarray
            Aggregated membership values.

        Returns
        -------
        float
            Bisector crisp value.

        Raises
        ------
        ValueError
            If the total area is zero.
        """
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

    # ------------------------------------------------------------------
    # Max membership (mean of maxima)
    # ------------------------------------------------------------------

    @staticmethod
    def _defuzzify_max_membership(universe, aggregated):
        """Mean-of-maxima defuzzification.

        Returns the mean of all universe points where the aggregated
        membership reaches its maximum.

        Parameters
        ----------
        universe : numpy.ndarray
            Discrete universe points.
        aggregated : numpy.ndarray
            Aggregated membership values.

        Returns
        -------
        float
            Mean-of-maxima crisp value.

        Raises
        ------
        ValueError
            If the total area is zero.
        """
        max_val = np.max(aggregated)
        if max_val == 0.0:
            raise ValueError(
                "Cannot defuzzify: aggregated membership area is zero "
                "(no rules fired or all firing strengths are zero)."
            )
        max_indices = np.where(aggregated == max_val)[0]
        return float(np.mean(universe[max_indices]))

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"FuzzyInferenceSystem("
            f"inputs={list(self.inputs.keys())}, "
            f"outputs={list(self.outputs.keys())}, "
            f"rules={len(self.rules)})"
        )
