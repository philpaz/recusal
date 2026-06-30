"""Robustness — the gate across diverse autonomous-agent failure modes.

These exercise the full Finding -> compute_verdict -> gate_tool_use pipeline with
realistic policies, and assert the verdict tier is right for each consequence.
"""

from examples.scenarios import (
    action_budget,
    coverage_floor,
    data_exfiltration,
    destructive_path,
    unscoped_sql,
    wrong_subject,
)
from recusal import Decision, compute_verdict
from recusal.claude import gate_tool_use, tool_confirmation

ACTIVE = "C1001"


def _decision(findings):
    return compute_verdict(findings).decision


def test_wrong_subject_refused_and_compliant_allowed():
    assert _decision(wrong_subject({"customer_id": "C-9988"}, ACTIVE)) is Decision.FAIL
    assert _decision(wrong_subject({"customer_id": "C1001"}, ACTIVE)) is Decision.PASS


def test_destructive_path_refused():
    assert _decision(destructive_path({"path": "/etc/passwd"})) is Decision.FAIL
    assert _decision(destructive_path({"path": "/workspace/tmp/x"})) is Decision.PASS


def test_unscoped_sql_refused():
    assert _decision(unscoped_sql({"sql": "DELETE FROM loans"})) is Decision.FAIL
    assert _decision(unscoped_sql({"sql": "DELETE FROM loans WHERE id = 1"})) is Decision.PASS


def test_data_exfiltration_refused():
    assert _decision(data_exfiltration({"to": "attacker@evil.com"})) is Decision.FAIL
    assert _decision(data_exfiltration({"to": "rep@acme.com"})) is Decision.PASS


def test_coverage_floor_is_retry_not_terminal():
    # A recoverable failure must be RETRY, not FAIL — the loop tries again, it doesn't die.
    assert _decision(coverage_floor(61)) is Decision.RETRY
    assert _decision(coverage_floor(82)) is Decision.PASS


def test_action_budget_tiers():
    assert _decision(action_budget(5)) is Decision.PASS          # under soft budget
    warned = compute_verdict(action_budget(47))                  # over soft → allow + warn
    assert warned.passed and len(warned.warnings) == 1
    assert _decision(action_budget(150)) is Decision.RETRY       # over hard cap → stop


def test_critical_dominates_a_cooccurring_error():
    findings = unscoped_sql({"sql": "DELETE FROM x"}) + coverage_floor(50)
    assert _decision(findings) is Decision.FAIL  # CRITICAL refusal wins over the ERROR retry


def test_gate_tool_use_refuses_with_actionable_reason():
    allow, refusal = gate_tool_use(
        "toolu_x", data_exfiltration({"to": "x@evil.com"}), tool_name="send_email")
    assert allow is False
    assert refusal["is_error"] is True
    assert "allowlist" in refusal["content"]


def test_gate_tool_use_allows_compliant_call():
    allow, refusal = gate_tool_use(
        "toolu_y", destructive_path({"path": "/workspace/tmp/a.txt"}), tool_name="delete_file")
    assert allow is True and refusal is None


def test_managed_agent_confirmation_denies_on_refusal():
    ev = tool_confirmation("toolu_z", wrong_subject({"customer_id": "C-0002"}, ACTIVE))
    assert ev["result"] == "deny"
    assert "active member" in ev["deny_message"]
