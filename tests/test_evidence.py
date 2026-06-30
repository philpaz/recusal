"""Hermetic tests for the evidence contract itself — Finding / Verdict / coercion."""

from recusal import Finding, Severity
from recusal.evidence import Decision


def test_finding_constructors():
    ok = Finding.ok("row_count", severity="INFO", actual=5)
    assert ok.passed and ok.severity is Severity.INFO and ok.context["actual"] == 5

    bad = Finding.fail("row_count", severity="CRITICAL", message="empty", actual=0)
    assert not bad.passed and bad.severity is Severity.CRITICAL


def test_coerce_from_dict_status_variants():
    assert Finding.coerce({"severity": "ERROR", "status": "fail"}).passed is False
    assert Finding.coerce({"severity": "WARNING", "status": "warn"}).passed is False
    assert Finding.coerce({"severity": "INFO", "status": "pass"}).passed is True
    # legacy 'type' maps to 'check'
    assert Finding.coerce({"type": "null_rate", "status": "pass"}).check == "null_rate"


def test_coerce_passthrough_and_typeerror():
    f = Finding.ok("c")
    assert Finding.coerce(f) is f
    try:
        Finding.coerce(42)  # type: ignore[arg-type]
        assert False, "expected TypeError"
    except TypeError:
        pass


def test_verdict_reasons_lists_failure_messages():
    from recusal import compute_verdict

    v = compute_verdict(
        [
            Finding.fail("a", severity="CRITICAL", message="thing one"),
            Finding.fail("b", severity="CRITICAL", message="thing two"),
        ]
    )
    assert v.decision is Decision.FAIL
    assert "thing one" in v.reasons() and "thing two" in v.reasons()
