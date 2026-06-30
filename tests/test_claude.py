"""Hermetic tests for the Claude adapter — no Anthropic SDK, no network."""

from recusal.claude import gate_tool_use, tool_confirmation

CRITICAL_FAIL = [{"severity": "CRITICAL", "status": "fail", "message": "would delete prod table"}]
CLEAN = [{"severity": "INFO", "status": "pass", "message": "dry-run ok"}]


def test_gate_allows_a_clean_tool_call():
    allow, refusal = gate_tool_use("toolu_1", CLEAN, tool_name="run_query")
    assert allow is True
    assert refusal is None


def test_gate_refuses_and_returns_is_error_tool_result():
    allow, refusal = gate_tool_use("toolu_2", CRITICAL_FAIL, tool_name="drop_table")
    assert allow is False
    assert refusal["type"] == "tool_result"
    assert refusal["tool_use_id"] == "toolu_2"
    assert refusal["is_error"] is True
    assert "drop_table" in refusal["content"]
    assert "delete prod table" in refusal["content"]


def test_retry_verdict_also_blocks():
    findings = [{"severity": "ERROR", "status": "fail", "message": "stale precondition"}]
    allow, refusal = gate_tool_use("toolu_3", findings)
    assert allow is False
    assert refusal["is_error"] is True


def test_tool_confirmation_allow():
    ev = tool_confirmation("toolu_4", CLEAN)
    assert ev["type"] == "user.tool_confirmation"
    assert ev["tool_use_id"] == "toolu_4"
    assert ev["result"] == "allow"
    assert "deny_message" not in ev


def test_tool_confirmation_deny_carries_reason():
    ev = tool_confirmation("toolu_5", CRITICAL_FAIL)
    assert ev["result"] == "deny"
    assert "delete prod table" in ev["deny_message"]
