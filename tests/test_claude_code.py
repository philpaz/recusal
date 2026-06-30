"""Hermetic tests for the Claude Code PreToolUse hook adapter — stdin/stdout JSON."""

import io
import json

from recusal import Finding
from recusal.claude_code import decide, run_pretooluse_hook


def _deny_rm(tool_name, tool_input):
    if tool_name == "Bash" and "rm -rf" in tool_input.get("command", ""):
        return [Finding.fail("destructive_bash", severity="CRITICAL", message="refusing rm -rf")]
    return []


def test_decide_defers_on_clean():
    decision, _ = decide("Read", {"file_path": "/x"}, _deny_rm)
    assert decision == "defer"  # no opinion → Claude Code's normal flow runs


def test_decide_denies_destructive():
    decision, reason = decide("Bash", {"command": "rm -rf /"}, _deny_rm)
    assert decision == "deny"
    assert "rm -rf" in reason


def test_decide_allow_on_pass_is_opt_in():
    decision, _ = decide("Read", {}, _deny_rm, allow_on_pass=True)
    assert decision == "allow"


def _run(event):
    out = io.StringIO()
    res = run_pretooluse_hook(_deny_rm, stdin=io.StringIO(json.dumps(event)), stdout=out)
    return res, out.getvalue()


def test_hook_emits_pretooluse_deny_json():
    _res, text = _run(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "rm -rf /"},
        }
    )
    hso = json.loads(text)["hookSpecificOutput"]
    assert hso["hookEventName"] == "PreToolUse"
    assert hso["permissionDecision"] == "deny"
    assert "rm -rf" in hso["permissionDecisionReason"]


def test_hook_defers_silently_on_clean():
    res, text = _run(
        {"hook_event_name": "PreToolUse", "tool_name": "Read", "tool_input": {"file_path": "/x"}}
    )
    assert res is None
    assert text == ""


def test_hook_survives_malformed_event():
    out = io.StringIO()
    res = run_pretooluse_hook(_deny_rm, stdin=io.StringIO("not json"), stdout=out)
    assert res is None  # empty event → no findings → defer, never crash the tool path
