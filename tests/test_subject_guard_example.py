"""Recipe 4: a write is not approved without a bound active subject."""

import pytest

from examples.subject_guard import make_subject_guard
from recusal import Decision, compute_verdict
from recusal.claude_code import decide


@pytest.mark.parametrize(
    ("active_id", "target", "expected"),
    [
        ("C1001", "C1001", Decision.PASS),
        ("C1001", "C2002", Decision.FAIL),
        ("C1001", None, Decision.FAIL),
        (None, None, Decision.FAIL),
        ("", "", Decision.FAIL),
        ("   ", "   ", Decision.FAIL),
        (False, False, Decision.FAIL),
        (42, 42, Decision.FAIL),
        ("C1001", "", Decision.FAIL),
    ],
)
def test_subject_guard_table(active_id, target, expected):
    findings = make_subject_guard(active_id)("update_record", {"id": target})
    assert compute_verdict(findings).decision is expected
    decision, _ = decide("update_record", {"id": target}, make_subject_guard(active_id))
    assert decision == ("defer" if expected is Decision.PASS else "deny")


@pytest.mark.parametrize("active_id", [None, "", "  ", False, 42])
def test_unbound_subject_refuses_with_actionable_reason(active_id):
    decision, reason = decide("update_record", {}, make_subject_guard(active_id))
    assert decision == "deny"
    assert "no active subject is bound" in reason
    evidence = make_subject_guard(active_id)("update_record", {})
    assert evidence[0].check == "subject_unbound"


def test_missing_target_with_valid_subject_refuses():
    evidence = make_subject_guard("C1001")("update_record", {})
    assert compute_verdict(evidence).decision is Decision.FAIL
    assert evidence[0].check == "subject_match"


def test_other_tool_calls_defer_to_host_permissions():
    assert make_subject_guard(None)("Read", {"file_path": "notes.md"}) == []
    assert decide("Read", {"file_path": "notes.md"}, make_subject_guard(None))[0] == "defer"
