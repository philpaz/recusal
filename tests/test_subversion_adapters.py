"""
Adversarial suite: try to get an *enforcement adapter* to allow a call it should block.

The adapters (`recusal.claude` and `recusal.claude_code`) are the seam between a verdict
and a real tool execution. Their one non-negotiable property is **fail closed**: a buggy
policy, ambiguous evidence, or a malformed event must resolve to a refusal, never a silent
allow. These tests hammer that boundary.
"""

import io
import json

from recusal import Finding
from recusal.claude import gate_tool_use, tool_confirmation
from recusal.claude_code import decide, run_pretooluse_hook

# --- objective: get a PASS/allow out of ambiguous or malformed evidence -----------------


class TestAdaptersFailClosedOnBadEvidence:
    def test_gate_tool_use_refuses_ambiguous_dict(self):
        # A CRITICAL-shaped dict with no status/passed must not become an allow at the seam.
        allow, block = gate_tool_use("t1", [{"severity": "CRITICAL", "message": "danger"}])
        assert allow is False
        assert block["is_error"] is True

    def test_gate_tool_use_refuses_stringified_false(self):
        allow, _ = gate_tool_use("t1", [{"severity": "CRITICAL", "passed": "false"}])
        assert allow is False

    def test_tool_confirmation_denies_ambiguous_dict(self):
        ev = tool_confirmation("t1", [{"severity": "CRITICAL", "message": "danger"}])
        assert ev["result"] == "deny"
        assert "deny_message" in ev

    def test_decide_uses_strict_boundary(self):
        # The Claude Code adapter adjudicates strict=True; an ambiguous finding is a deny.
        d, _ = decide("Bash", {"command": "x"}, lambda n, i: [{"severity": "CRITICAL"}])
        assert d == "deny"

    def test_unknown_severity_fails_closed_everywhere(self):
        bad = [{"severity": "APOCALYPTIC", "status": "fail"}]
        assert gate_tool_use("t", bad)[0] is False
        assert tool_confirmation("t", bad)["result"] == "deny"
        assert decide("Bash", {"command": "x"}, lambda n, i: bad)[0] == "deny"


# --- objective: disable the gate by throwing from the policy ----------------------------


class TestRaisingPolicyFailsClosed:
    def _boom(self, *_):
        raise RuntimeError("policy blew up")

    def test_decide_denies_when_policy_raises(self):
        d, reason = decide("Bash", {"command": "rm -rf /"}, self._boom)
        assert d == "deny"
        assert "failed closed" in reason.lower()

    def test_fail_open_is_opt_in_only(self):
        # You can *choose* to fail open, but it is never the default.
        d, _ = decide("Bash", {"command": "x"}, self._boom, fail_closed=False)
        assert d == "defer"

    def test_policy_returning_none_is_treated_as_no_findings(self):
        d, _ = decide("Bash", {"command": "x"}, lambda n, i: None)
        assert d == "defer"  # PASS -> defer, never a spurious allow


# --- objective: slip a malformed hook envelope past the gate ----------------------------


def _run(stdin_text, policy, **kw):
    out = io.StringIO()
    result = run_pretooluse_hook(policy, stdin=io.StringIO(stdin_text), stdout=out, **kw)
    return result, out.getvalue()


class TestMalformedEnvelopeFailsClosed:
    _allow_all = staticmethod(lambda n, i: [])  # a policy that never objects

    def test_non_json_stdin_denies(self):
        result, _ = _run("this is not json", self._allow_all)
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_json_that_is_not_an_object_denies(self):
        for payload in ("[]", "42", '"a string"', "null"):
            result, _ = _run(payload, self._allow_all)
            assert result["hookSpecificOutput"]["permissionDecision"] == "deny", payload

    def test_missing_tool_name_denies(self):
        result, _ = _run(json.dumps({"tool_input": {"command": "x"}}), self._allow_all)
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_non_dict_tool_input_denies(self):
        event = json.dumps({"tool_name": "Bash", "tool_input": "not-a-dict"})
        result, _ = _run(event, self._allow_all)
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_truncated_json_denies(self):
        result, _ = _run('{"tool_name": "Bash", "tool_input": {"comm', self._allow_all)
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_well_formed_pass_defers_and_emits_nothing(self):
        event = json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls"}})
        result, out = _run(event, self._allow_all)
        assert result is None  # defer
        assert out == ""  # no JSON written -> Claude Code's normal flow runs

    def test_malformed_event_can_opt_into_fail_open(self):
        result, _ = _run("garbage", self._allow_all, fail_closed=False)
        assert result is None  # explicit opt-out defers instead of denying


# --- objective: force an auto-allow that bypasses the permission prompt ------------------


class TestAllowOnPassIsNeverImplicit:
    def test_pass_defers_not_allows_by_default(self):
        d, _ = decide("Bash", {"command": "ls"}, lambda n, i: [Finding.ok("fine")])
        assert d == "defer"  # a gate adds refusals; it does not strip the prompt

    def test_allow_on_pass_must_be_explicit(self):
        d, _ = decide(
            "Bash", {"command": "ls"}, lambda n, i: [Finding.ok("fine")], allow_on_pass=True
        )
        assert d == "allow"

    def test_allow_on_pass_still_cannot_flip_a_failure(self):
        # Even in auto-approve mode, a FAIL is a deny - allow_on_pass only affects PASS.
        d, _ = decide(
            "Bash",
            {"command": "x"},
            lambda n, i: [Finding.fail("no", severity="CRITICAL")],
            allow_on_pass=True,
        )
        assert d == "deny"
