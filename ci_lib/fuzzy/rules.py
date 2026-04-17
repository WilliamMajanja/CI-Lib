"""Fuzzy rules for inference systems."""

from ci_lib.fuzzy.variables import FuzzyVariable


class FuzzyRule:
    """A single IF-THEN fuzzy rule.

    Parameters
    ----------
    antecedents : list of (FuzzyVariable, str)
        Each tuple pairs an input variable with the name of one of its
        registered fuzzy sets.
    consequent : (FuzzyVariable, str)
        A tuple pairing an output variable with the name of one of its
        registered fuzzy sets.
    connective : {'and', 'or'}, optional
        How multiple antecedents are combined.  ``'and'`` uses *min*,
        ``'or'`` uses *max*.  Default is ``'and'``.

    Attributes
    ----------
    antecedents : list of (FuzzyVariable, str)
        The rule's antecedent pairs.
    consequent : (FuzzyVariable, str)
        The rule's consequent pair.
    connective : str
        ``'and'`` or ``'or'``.

    Raises
    ------
    TypeError
        If *antecedents* or *consequent* have the wrong structure.
    ValueError
        If *connective* is not ``'and'`` or ``'or'``, or a referenced
        fuzzy-set name is not registered on its variable.

    Examples
    --------
    >>> rule = FuzzyRule(
    ...     [(temp_var, "hot"), (humidity_var, "high")],
    ...     (fan_var, "fast"),
    ...     connective="and",
    ... )
    >>> rule.evaluate({"temperature": 75, "humidity": 80})
    0.6
    """

    _CONNECTIVES = {"and", "or"}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, antecedents, consequent, connective="and"):
        # --- connective ---------------------------------------------------
        if connective not in self._CONNECTIVES:
            raise ValueError(
                f"connective must be 'and' or 'or', got {connective!r}"
            )

        # --- antecedents --------------------------------------------------
        if not isinstance(antecedents, list) or len(antecedents) == 0:
            raise TypeError("antecedents must be a non-empty list of tuples")
        for item in antecedents:
            self._validate_var_set_pair(item, "antecedent")

        # --- consequent ---------------------------------------------------
        self._validate_var_set_pair(consequent, "consequent")

        self.antecedents = antecedents
        self.consequent = consequent
        self.connective = connective

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_var_set_pair(pair, label):
        """Check that *pair* is ``(FuzzyVariable, set_name_str)``.

        Parameters
        ----------
        pair : tuple
            The pair to validate.
        label : str
            Human-readable label used in error messages.

        Raises
        ------
        TypeError
            If *pair* is not a 2-tuple of ``(FuzzyVariable, str)``.
        ValueError
            If the set name is not registered on the variable.
        """
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(f"Each {label} must be a (FuzzyVariable, str) tuple")
        var, set_name = pair
        if not isinstance(var, FuzzyVariable):
            raise TypeError(
                f"{label} variable must be a FuzzyVariable, "
                f"got {type(var).__name__}"
            )
        if not isinstance(set_name, str):
            raise TypeError(
                f"{label} set name must be a string, "
                f"got {type(set_name).__name__}"
            )
        if set_name not in var.sets:
            raise ValueError(
                f"Set {set_name!r} not found in variable {var.name!r}. "
                f"Available sets: {list(var.sets.keys())}"
            )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, inputs):
        """Compute the firing strength of this rule.

        Parameters
        ----------
        inputs : dict
            Mapping ``{variable_name: crisp_value}`` for every input
            variable referenced in the antecedents.

        Returns
        -------
        float
            Firing strength in [0, 1].

        Raises
        ------
        KeyError
            If *inputs* is missing a required variable name.
        """
        degrees = []
        for var, set_name in self.antecedents:
            if var.name not in inputs:
                raise KeyError(
                    f"Missing input for variable {var.name!r}"
                )
            crisp = inputs[var.name]
            degrees.append(var.sets[set_name].degree(crisp))

        if self.connective == "and":
            return float(min(degrees))
        return float(max(degrees))

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self):
        ant_str = " {} ".format(self.connective.upper()).join(
            f"{v.name} IS {s}" for v, s in self.antecedents
        )
        con_var, con_set = self.consequent
        return f"FuzzyRule(IF {ant_str} THEN {con_var.name} IS {con_set})"
