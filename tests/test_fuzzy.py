"""Tests for ci_lib.fuzzy (FuzzySet, FuzzyVariable, FuzzyRule, FuzzyInferenceSystem)."""

import numpy as np
import pytest

from ci_lib.fuzzy.sets import FuzzySet
from ci_lib.fuzzy.variables import FuzzyVariable
from ci_lib.fuzzy.rules import FuzzyRule
from ci_lib.fuzzy.inference import FuzzyInferenceSystem


# ------------------------------------------------------------------
# FuzzySet membership functions
# ------------------------------------------------------------------

class TestTriangular:
    def test_peak_is_one(self):
        fs = FuzzySet.triangular("med", 2, 5, 8)
        assert fs.degree(5) == pytest.approx(1.0)

    def test_outside_is_zero(self):
        fs = FuzzySet.triangular("med", 2, 5, 8)
        assert fs.degree(0) == pytest.approx(0.0)
        assert fs.degree(10) == pytest.approx(0.0)

    def test_midpoint(self):
        fs = FuzzySet.triangular("med", 0, 5, 10)
        assert fs.degree(2.5) == pytest.approx(0.5)


class TestTrapezoidal:
    def test_plateau_is_one(self):
        fs = FuzzySet.trapezoidal("hi", 2, 4, 6, 8)
        assert fs.degree(5) == pytest.approx(1.0)

    def test_outside_is_zero(self):
        fs = FuzzySet.trapezoidal("hi", 2, 4, 6, 8)
        assert fs.degree(0) == pytest.approx(0.0)
        assert fs.degree(10) == pytest.approx(0.0)


class TestGaussian:
    def test_at_mean_is_one(self):
        fs = FuzzySet.gaussian("g", mean=5.0, sigma=2.0)
        assert fs.degree(5.0) == pytest.approx(1.0)

    def test_away_from_mean_less_than_one(self):
        fs = FuzzySet.gaussian("g", mean=5.0, sigma=2.0)
        assert fs.degree(10.0) < 1.0


# ------------------------------------------------------------------
# FuzzyVariable
# ------------------------------------------------------------------

class TestFuzzyVariable:
    def test_fuzzify_returns_dict(self):
        var = FuzzyVariable("temp", (0, 100))
        var.add_set("low", FuzzySet.triangular("low", 0, 0, 50))
        var.add_set("high", FuzzySet.triangular("high", 50, 100, 100))
        result = var.fuzzify(25)
        assert isinstance(result, dict)
        assert "low" in result and "high" in result


# ------------------------------------------------------------------
# FuzzyRule
# ------------------------------------------------------------------

class TestFuzzyRule:
    def test_evaluate_and_connective(self):
        food = FuzzyVariable("food", (0, 10))
        food.add_set("good", FuzzySet.triangular("good", 5, 10, 10))

        service = FuzzyVariable("service", (0, 10))
        service.add_set("good", FuzzySet.triangular("good", 5, 10, 10))

        tip = FuzzyVariable("tip", (0, 30))
        tip.add_set("high", FuzzySet.triangular("high", 15, 25, 30))

        rule = FuzzyRule(
            [(food, "good"), (service, "good")],
            (tip, "high"),
            connective="and",
        )
        strength = rule.evaluate({"food": 8, "service": 6})
        assert 0.0 <= strength <= 1.0


# ------------------------------------------------------------------
# FuzzyInferenceSystem (simple tipping problem)
# ------------------------------------------------------------------

class TestFuzzyInferenceSystem:
    @pytest.fixture
    def tipping_system(self):
        food = FuzzyVariable("food", (0, 10), resolution=50)
        food.add_set("bad", FuzzySet.triangular("bad", 0, 0, 5))
        food.add_set("good", FuzzySet.triangular("good", 5, 10, 10))

        service = FuzzyVariable("service", (0, 10), resolution=50)
        service.add_set("poor", FuzzySet.triangular("poor", 0, 0, 5))
        service.add_set("great", FuzzySet.triangular("great", 5, 10, 10))

        tip = FuzzyVariable("tip", (0, 30), resolution=50)
        tip.add_set("low", FuzzySet.triangular("low", 0, 5, 15))
        tip.add_set("high", FuzzySet.triangular("high", 15, 25, 30))

        rule1 = FuzzyRule(
            [(food, "bad"), (service, "poor")],
            (tip, "low"), connective="and",
        )
        rule2 = FuzzyRule(
            [(food, "good"), (service, "great")],
            (tip, "high"), connective="and",
        )

        fis = FuzzyInferenceSystem()
        fis.add_variable(food, var_type="input")
        fis.add_variable(service, var_type="input")
        fis.add_variable(tip, var_type="output")
        fis.add_rule(rule1)
        fis.add_rule(rule2)
        return fis

    def test_compute_returns_dict(self, tipping_system):
        result = tipping_system.compute({"food": 8, "service": 9})
        assert isinstance(result, dict)
        assert "tip" in result

    def test_good_input_gives_higher_tip(self, tipping_system):
        good = tipping_system.compute({"food": 9, "service": 9})
        bad = tipping_system.compute({"food": 1, "service": 1})
        assert good["tip"] > bad["tip"]
