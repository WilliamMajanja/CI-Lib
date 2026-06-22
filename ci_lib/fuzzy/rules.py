"""Fuzzy rules for inference systems."""

from __future__ import annotations

from ci_lib.fuzzy.variables import FuzzyVariable


class FuzzyRule:
    """A single IF-THEN fuzzy rule.

    Parameters
    ----------
    antecedents : list of (FuzzyVariable, str)
        Each tuple pairs an input variable with the name of one of its
        registered fuzzy sets.
    consequent : (FuzzyVariable, str)
        Tuple pairing an output variable with one of its fuzzy sets.
    connective : {'and', 'or'}, optional
        How multiple antecedents are combined (default ``'and'``).

    References
    ----------
    .. [1] Mamdani, E. H., & Assilian, S. (1975). An experiment in linguistic
           synthesis with a fuzzy logic controller. Int. J. Man-Machine Studies.
    .. [2] Takagi, T., & Sugeno, M. (1985). Fuzzy identification of systems
           and its applications. IEEE Trans. SMC.
    """

    _CONNECTIVES = {"and", "or"}

    def __init__(
        self,
        antecedents: list[tuple[FuzzyVariable, str]],
        consequent: tuple[FuzzyVariable, str],
        connective: str = "and",
    ) -> None:
        if connective not in self._CONNECTIVES:
            raise ValueError(f"connective must be 'and' or 'or', got {connective!r}")

        if not isinstance(antecedents, list) or len(antecedents) == 0:
            raise TypeError("antecedents must be a non-empty list of tuples")
        for item in antecedents:
            self._validate_var_set_pair(item, "antecedent")

        self._validate_var_set_pair(consequent, "consequent")

        self.antecedents = antecedents
        self.consequent = consequent
        self.connective = connective

    @staticmethod
    def _validate_var_set_pair(pair: tuple, label: str) -> None:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(f"Each {label} must be a (FuzzyVariable, str) tuple")
        var, set_name = pair
        if not isinstance(var, FuzzyVariable):
            raise TypeError(
                f"{label} variable must be a FuzzyVariable, got {type(var).__name__}"
            )
        if not isinstance(set_name, str):
            raise TypeError(
                f"{label} set name must be a string, got {type(set_name).__name__}"
            )
        if set_name not in var.sets:
            raise ValueError(
                f"Set {set_name!r} not found in variable {var.name!r}. "
                f"Available sets: {list(var.sets.keys())}"
            )

    def evaluate(self, inputs: dict[str, float]) -> float:
        degrees = []
        for var, set_name in self.antecedents:
            if var.name not in inputs:
                raise KeyError(f"Missing input for variable {var.name!r}")
            crisp = inputs[var.name]
            degrees.append(var.sets[set_name].degree(crisp))

        if self.connective == "and":
            return float(min(degrees))
        return float(max(degrees))

    def __repr__(self) -> str:
        ant_str = " {} ".format(self.connective.upper()).join(
            f"{v.name} IS {s}" for v, s in self.antecedents
        )
        con_var, con_set = self.consequent
        return f"FuzzyRule(IF {ant_str} THEN {con_var.name} IS {con_set})"
