"""Recipe 8: bad CI measurements must not be treated as passing evidence."""

import math

import pytest

from examples.quality_gate import quality_gate
from recusal import Decision, compute_verdict


@pytest.mark.parametrize(
    ("coverage", "failed", "expected"),
    [
        (61, 2, Decision.RETRY),
        (80, 0, Decision.PASS),
        (75, 0, Decision.PASS),
        (74.9, 0, Decision.RETRY),
        (math.nan, 0, Decision.RETRY),
        (80, math.nan, Decision.RETRY),
        (80, -1, Decision.RETRY),
        (None, 0, Decision.RETRY),
        ("80", 0, Decision.RETRY),
        (True, 0, Decision.RETRY),
        (80, True, Decision.RETRY),
        (80, 0.0, Decision.RETRY),
        (math.inf, 0, Decision.RETRY),
        (-math.inf, 0, Decision.RETRY),
        (101, 0, Decision.RETRY),
        (-1, 0, Decision.RETRY),
        (80, None, Decision.RETRY),
        (80, "0", Decision.RETRY),
    ],
)
def test_recipe8_contract(coverage, failed, expected):
    assert compute_verdict(quality_gate(coverage, failed)).decision is expected


@pytest.mark.parametrize(
    ("coverage", "failed", "minimum", "check"),
    [
        (math.nan, 0, 75, "coverage_input"),
        (None, 0, 75, "coverage_input"),
        ("80", 0, 75, "coverage_input"),
        (80, math.nan, 75, "tests_input"),
        (80, -1, 75, "tests_input"),
        (80, False, 75, "tests_input"),
        (80, 0, math.nan, "minimum_coverage_input"),
        (80, 0, "75", "minimum_coverage_input"),
        (80, 0, True, "minimum_coverage_input"),
    ],
)
def test_unreadable_input_has_named_error_finding(coverage, failed, minimum, check):
    evidence = quality_gate(coverage, failed, min_coverage=minimum)
    assert compute_verdict(evidence).decision is Decision.RETRY
    assert any(f.check == check and f.message for f in evidence)


def test_both_known_recoverable_failures_are_retained():
    evidence = quality_gate(61, 2)
    assert [f.check for f in evidence] == ["coverage_floor", "tests"]
    assert compute_verdict(evidence).decision is Decision.RETRY


def test_multiple_unusable_measurements_are_retained():
    evidence = quality_gate(None, math.nan, min_coverage="")
    assert {f.check for f in evidence} == {
        "coverage_input",
        "minimum_coverage_input",
        "tests_input",
    }
    assert compute_verdict(evidence).decision is Decision.RETRY


def test_extreme_integer_coverage_does_not_crash():
    evidence = quality_gate(10**1000, 0)
    assert compute_verdict(evidence).decision is Decision.RETRY


def test_example_function_matches_the_cookbook_default_floor():
    assert quality_gate(74, 0)[0].check == "coverage_floor"
    assert quality_gate(75, 0) == []
