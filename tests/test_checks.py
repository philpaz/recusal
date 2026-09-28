"""Hermetic tests for the built-in checks, plain dicts, no pandas. Checks emit Findings."""

from recusal import compute_verdict
from recusal.checks import (
    date_range,
    in_range,
    in_set,
    null_rate,
    referential_integrity,
    required_keys,
    row_count,
)

USERS = [
    {"id": 1, "email": "a@x.com", "score": 70},
    {"id": 2, "email": "", "score": 95},
    {"id": 3, "email": "c@x.com", "score": 40},
]
ORDERS = [
    {"id": 10, "user_id": 1, "plan": "free"},
    {"id": 11, "user_id": 99, "plan": "pro"},  # orphan
    {"id": 12, "user_id": 3, "plan": "wildcat"},  # not in approved set
]


def test_row_count_pass_and_fail():
    assert row_count(USERS, min_rows=1).passed
    fail = row_count([], min_rows=1)
    assert not fail.passed
    assert fail.severity == "CRITICAL"


def test_row_count_rejects_nan_threshold():
    finding = row_count([], min_rows=float("nan"))
    assert not finding.passed
    assert "invalid threshold" in finding.message


def test_row_count_rejects_positive_infinity_threshold():
    finding = row_count([], min_rows=float("inf"))
    assert not finding.passed
    assert "invalid threshold" in finding.message


def test_row_count_rejects_negative_infinity_threshold():
    finding = row_count([], min_rows=float("-inf"))
    assert not finding.passed
    assert "invalid threshold" in finding.message


def test_row_count_rejects_bool_threshold():
    finding = row_count([], min_rows=True)
    assert not finding.passed
    assert "invalid threshold" in finding.message


def test_row_count_rejects_negative_threshold():
    finding = row_count([], min_rows=-1)
    assert not finding.passed
    assert "invalid threshold" in finding.message


def test_null_rate_flags_empty_string():
    f = null_rate(USERS, "email", max_rate=0.10)  # 1/3 = 33% > 10%
    assert not f.passed
    assert "33" in f.message or "0.33" in str(f.context["null_rate"])


def test_null_rate_rejects_nan_threshold():
    f = null_rate(USERS, "email", max_rate=float("nan"))
    assert not f.passed
    assert "invalid threshold" in f.message


def test_null_rate_rejects_inf_threshold():
    f = null_rate(USERS, "email", max_rate=float("inf"))
    assert not f.passed
    assert "invalid threshold" in f.message


def test_null_rate_rejects_bool_threshold():
    f = null_rate(USERS, "email", max_rate=True)
    assert not f.passed
    assert "invalid threshold" in f.message


def test_null_rate_rejects_rate_above_1():
    f = null_rate(USERS, "email", max_rate=1.5)
    assert not f.passed
    assert "invalid threshold" in f.message


def test_null_rate_rejects_negative_rate():
    f = null_rate(USERS, "email", max_rate=-0.1)
    assert not f.passed
    assert "invalid threshold" in f.message


def test_referential_integrity_finds_orphan():
    f = referential_integrity(ORDERS, USERS, fk="user_id", pk="id")
    assert not f.passed
    assert f.context["orphan_count"] == 1


def test_in_set_flags_unapproved_value():
    f = in_set(ORDERS, "plan", allowed=["free", "pro"])
    assert not f.passed
    assert f.context["violation_count"] == 1


def test_in_range_flags_out_of_bounds():
    assert in_range(USERS, "score", min_value=0, max_value=100).passed
    f2 = in_range(USERS, "score", min_value=50, max_value=100)
    assert not f2.passed
    assert f2.context["violation_count"] == 1


def test_in_range_rejects_nan_min():
    f = in_range(USERS, "score", min_value=float("nan"), max_value=100)
    assert not f.passed
    assert "invalid threshold" in f.message


def test_in_range_rejects_nan_max():
    f = in_range(USERS, "score", min_value=0, max_value=float("nan"))
    assert not f.passed
    assert "invalid threshold" in f.message


def test_in_range_rejects_bool_threshold():
    f = in_range(USERS, "score", min_value=True, max_value=100)
    assert not f.passed
    assert "invalid threshold" in f.message


def test_in_range_rejects_inverted_bounds():
    f = in_range(USERS, "score", min_value=100, max_value=50)
    assert not f.passed
    assert "invalid threshold" in f.message


def test_date_range_pass_and_fail():
    dates = [
        {"id": 1, "created": "2026-01-05"},
        {"id": 2, "created": "2026-01-15T12:00:00"},
        {"id": 3, "created": "2026-01-20"},
    ]
    assert date_range(dates, "created", min_date="2026-01-01", max_date="2026-01-31").passed
    f = date_range(dates, "created", min_date="2026-01-10", max_date="2026-01-31")
    assert not f.passed
    assert f.context["violation_count"] == 1


def test_required_keys_detects_missing():
    rows = [{"id": 1, "email": "a"}, {"id": 2}]
    f = required_keys(rows, keys=["id", "email"])
    assert not f.passed
    assert "email" in f.context["missing_keys"]


def test_checks_compose_into_a_verdict():
    findings = [
        row_count(USERS, min_rows=1),
        referential_integrity(ORDERS, USERS, fk="user_id", pk="id"),  # CRITICAL fail
        in_set(ORDERS, "plan", allowed=["free", "pro"]),  # ERROR fail
    ]
    verdict = compute_verdict(findings)
    assert verdict.refused  # CRITICAL orphan dominates
