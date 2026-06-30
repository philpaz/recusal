"""Heavy invariant tests for the evidence contract, precedence, idempotence, determinism."""

import itertools

import pytest

from recusal import Decision, Finding, Severity, compute_verdict


def _fail(sev):
    return Finding.fail("c", severity=sev, message=sev.lower())


def _ok(sev):
    return Finding.ok("c", severity=sev)


def test_critical_always_dominates_any_combination():
    for err, warn, info in itertools.product([0, 1], repeat=3):
        findings = [_fail("CRITICAL")]
        if err:
            findings.append(_fail("ERROR"))
        if warn:
            findings.append(_fail("WARNING"))
        if info:
            findings.append(_ok("INFO"))
        assert compute_verdict(findings).decision is Decision.FAIL


def test_error_dominates_warning_and_info():
    v = compute_verdict([_fail("ERROR"), _fail("WARNING"), _ok("INFO")])
    assert v.decision is Decision.RETRY


def test_only_warnings_pass_but_are_recorded():
    v = compute_verdict([_fail("WARNING"), _fail("WARNING")])
    assert v.passed
    assert len(v.warnings) == 2


def test_failed_info_is_a_metric_never_blocks():
    v = compute_verdict([Finding.fail("c", severity="INFO", message="weird")])
    assert v.passed
    assert len(v.metrics) == 1


def test_passing_or_info_findings_never_change_the_decision():
    base = [_fail("ERROR")]
    noise = [
        _ok("INFO"),
        _ok("CRITICAL"),
        _ok("ERROR"),
        _ok("WARNING"),
        Finding.fail("c", severity="INFO"),
    ]
    assert (
        compute_verdict(base).decision is compute_verdict(base + noise).decision is Decision.RETRY
    )


def test_decision_is_order_independent():
    findings = [_fail("WARNING"), _fail("CRITICAL"), _ok("INFO"), _fail("ERROR")]
    for perm in itertools.permutations(findings):
        assert compute_verdict(list(perm)).decision is Decision.FAIL


def test_determinism_and_counts_at_scale():
    findings = [_fail("ERROR") for _ in range(50)] + [_fail("WARNING") for _ in range(50)]
    a, b = compute_verdict(findings), compute_verdict(findings)
    assert a == b
    assert a.decision is Decision.RETRY
    assert len(a.failures) == 50 and len(a.warnings) == 50


def test_empty_evidence_is_info_pass():
    v = compute_verdict([])
    assert v.passed and v.highest_severity == "INFO" and v.failures == ()


def test_coerce_defaults_missing_fields():
    f = Finding.coerce({})  # no severity, no status
    assert f.severity is Severity.INFO and f.passed is True


def test_unknown_severity_in_dict_raises_clearly():
    with pytest.raises(ValueError):
        compute_verdict([{"severity": "BOGUS", "status": "fail"}])


def test_reasons_falls_back_to_summary_when_no_messages():
    v = compute_verdict([Finding.fail("c", severity="CRITICAL")])  # no message
    assert v.reasons() == "c"  # falls back to the check name, then summary


def test_mixed_finding_objects_and_dicts():
    v = compute_verdict(
        [
            Finding.fail("a", severity="CRITICAL", message="obj"),
            {"severity": "ERROR", "status": "fail", "message": "dict"},
        ]
    )
    assert v.decision is Decision.FAIL  # the CRITICAL Finding dominates
