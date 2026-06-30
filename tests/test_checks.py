"""Hermetic tests for the built-in checks — plain dicts, no pandas. Checks emit Findings."""

from recusal import compute_verdict
from recusal.checks import (
    in_range,
    in_set,
    null_rate,
    referential_integrity,
    required_keys,
    row_count,
)

MEMBERS = [
    {"id": 1, "email": "a@x.com", "score": 700},
    {"id": 2, "email": "", "score": 820},
    {"id": 3, "email": "c@x.com", "score": 410},
]
ACCOUNTS = [
    {"id": 10, "member_id": 1, "type": "checking"},
    {"id": 11, "member_id": 99, "type": "savings"},  # orphan
    {"id": 12, "member_id": 3, "type": "wildcat"},  # not in approved set
]


def test_row_count_pass_and_fail():
    assert row_count(MEMBERS, min_rows=1).passed
    fail = row_count([], min_rows=1)
    assert not fail.passed
    assert fail.severity == "CRITICAL"


def test_null_rate_flags_empty_string():
    f = null_rate(MEMBERS, "email", max_rate=0.10)  # 1/3 = 33% > 10%
    assert not f.passed
    assert "33" in f.message or "0.33" in str(f.context["null_rate"])


def test_referential_integrity_finds_orphan():
    f = referential_integrity(ACCOUNTS, MEMBERS, fk="member_id", pk="id")
    assert not f.passed
    assert f.context["orphan_count"] == 1


def test_in_set_flags_unapproved_value():
    f = in_set(ACCOUNTS, "type", allowed=["checking", "savings"])
    assert not f.passed
    assert f.context["violation_count"] == 1


def test_in_range_flags_out_of_bounds():
    assert in_range(MEMBERS, "score", min_value=300, max_value=850).passed
    f2 = in_range(MEMBERS, "score", min_value=500, max_value=850)
    assert not f2.passed
    assert f2.context["violation_count"] == 1


def test_required_keys_detects_missing():
    rows = [{"id": 1, "email": "a"}, {"id": 2}]
    f = required_keys(rows, keys=["id", "email"])
    assert not f.passed
    assert "email" in f.context["missing_keys"]


def test_checks_compose_into_a_verdict():
    findings = [
        row_count(MEMBERS, min_rows=1),
        referential_integrity(ACCOUNTS, MEMBERS, fk="member_id", pk="id"),  # CRITICAL fail
        in_set(ACCOUNTS, "type", allowed=["checking", "savings"]),  # ERROR fail
    ]
    verdict = compute_verdict(findings)
    assert verdict.refused  # CRITICAL orphan dominates
