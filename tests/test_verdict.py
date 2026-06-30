"""Hermetic tests for the tiered-severity verdict kernel (typed contract)."""

import dataclasses

import pytest

from recusal import Decision, Finding, Severity, compute_verdict


def test_critical_failure_is_terminal_fail():
    v = compute_verdict([
        Finding.fail("rows", severity="CRITICAL", message="0 rows in members"),
        Finding.fail("nulls", severity="ERROR", message="null rate high"),
    ])
    assert v.decision is Decision.FAIL
    assert v.refused
    assert v.highest_severity == "CRITICAL"
    assert len(v.failures) == 1  # only the CRITICAL drives the FAIL bucket
    assert "no retry" in v.message.lower()


def test_error_failure_is_retry():
    v = compute_verdict([Finding.fail("code", severity="ERROR", message="invalid product code")])
    assert v.decision is Decision.RETRY
    assert v.retryable
    assert "retry once" in v.message.lower()


def test_warnings_pass_but_are_recorded():
    v = compute_verdict([
        Finding.fail("dist", severity="WARNING", message="distribution drift"),
        Finding.ok("count", severity="INFO", value=1200),
    ])
    assert v.passed
    assert v.highest_severity == "WARNING"
    assert len(v.warnings) == 1
    assert len(v.metrics) == 1


def test_all_clear_is_info_pass():
    v = compute_verdict([Finding.ok("count", severity="INFO", value=5)])
    assert v.passed
    assert v.highest_severity == "INFO"


def test_empty_evidence_passes_vacuously():
    assert compute_verdict([]).passed


def test_accepts_loose_dicts_and_coerces():
    v = compute_verdict([{"severity": "CRITICAL", "status": "fail", "message": "boom"}])
    assert v.refused
    assert v.reasons() == "boom"


def test_severity_enum_compares_as_string():
    assert Severity.CRITICAL == "CRITICAL"
    assert Severity.ERROR.value == "ERROR"


def test_determinism_same_evidence_same_verdict():
    ev = [
        Finding.fail("a", severity="ERROR", message="x"),
        Finding.fail("b", severity="WARNING", message="y"),
    ]
    assert compute_verdict(ev) == compute_verdict(ev)


def test_finding_is_frozen():
    f = Finding.ok("c")
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.passed = False  # type: ignore[misc]
