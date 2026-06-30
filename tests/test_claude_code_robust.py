"""Heavy robustness tests for the Claude Code hook, fail-closed, malformed input, edge cases."""

import io
import json

from recusal import Finding
from recusal.claude_code import decide, run_pretooluse_hook


def _boom(tool_name, tool_input):
    raise RuntimeError("policy bug")


def _empty(tool_name, tool_input):
    return []


def _always_deny(tool_name, tool_input):
    return [Finding.fail("x", severity="CRITICAL", message="nope")]


def test_policy_error_fails_closed_by_default():
    decision, reason = decide("Bash", {}, _boom)
    assert decision == "deny"
    assert "failed closed" in reason.lower()


def test_policy_error_fail_open_is_opt_in():
    decision, _ = decide("Bash", {}, _boom, fail_closed=False)
    assert decision == "defer"


def test_bad_severity_in_policy_also_fails_closed():
    bad = lambda tn, ti: [{"severity": "NONSENSE", "status": "fail"}]  # noqa: E731
    decision, _ = decide("Bash", {}, bad)
    assert decision == "deny"


def test_empty_policy_defers():
    assert decide("Read", {}, _empty)[0] == "defer"


def test_retry_verdict_maps_to_deny():
    retry = lambda tn, ti: [Finding.fail("x", severity="ERROR", message="retry")]  # noqa: E731
    assert decide("Bash", {}, retry)[0] == "deny"


def _run(policy, event, **kw):
    out = io.StringIO()
    res = run_pretooluse_hook(
        policy,
        stdin=io.StringIO(event if isinstance(event, str) else json.dumps(event)),
        stdout=out,
        **kw,
    )
    return res, out.getvalue()


def test_hook_emits_deny_on_policy_error():
    _res, text = _run(_boom, {"tool_name": "Bash", "tool_input": {}})
    assert json.loads(text)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_hook_malformed_stdin_fails_closed():
    res, text = _run(_empty, "{not valid json")
    assert res["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "malformed" in text.lower()


def test_hook_non_object_json_fails_closed():
    # valid JSON but not an object (42, "x", [1,2], true, null) must not crash or fail open
    for payload in ("42", '"x"', "[1,2]", "true", "null"):
        res, text = _run(_empty, payload)
        assert res["hookSpecificOutput"]["permissionDecision"] == "deny", payload


def test_hook_missing_tool_fields_still_evaluated():
    _res, text = _run(_always_deny, {})  # no tool_name / tool_input
    assert json.loads(text)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_hook_allow_on_pass_opt_in_emits_allow():
    _res, text = _run(_empty, {"tool_name": "Read", "tool_input": {}}, allow_on_pass=True)
    assert json.loads(text)["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_hook_output_shape_is_exact():
    _res, text = _run(_always_deny, {"tool_name": "Bash", "tool_input": {"command": "x"}})
    payload = json.loads(text)
    assert set(payload) == {"hookSpecificOutput"}
    hso = payload["hookSpecificOutput"]
    assert hso["hookEventName"] == "PreToolUse"
    assert hso["permissionDecision"] in {"allow", "deny"}
    assert isinstance(hso["permissionDecisionReason"], str)
