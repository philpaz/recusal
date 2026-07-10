"""MCP tool calls are governed by the same PreToolUse gate as native tools.

Claude Code's hooks reference documents that MCP server tools "appear as regular tools
in tool events" (``PreToolUse``, ``PostToolUse``, ...) under the naming pattern
``mcp__<server>__<tool>`` (code.claude.com/docs/en/hooks). These tests pin what the
README claims about that: the ``policy(tool_name, tool_input)`` seam governs MCP calls
with no MCP-specific adapter, a clean MCP call *defers* (never auto-allows), allowlist
mode refuses MCP tools unless affirmatively named, and a buggy policy still fails closed
on an MCP event. See also ``examples/mcp_governance.py`` and cookbook recipe 12.
"""

import io
import json

from recusal import Finding
from recusal.claude_code import allowlist_policy, decide, run_pretooluse_hook

APPROVED_REPOS = frozenset({"philpaz/recusal"})


def mcp_policy(tool_name, tool_input):
    """The README's MCP snippet: refuse a destructive MCP action, scope a repo write,
    defer everything else."""
    if tool_name == "mcp__salesforce__delete_records":
        return [
            Finding.fail(
                "mcp_destructive_action",
                severity="CRITICAL",
                message="bulk Salesforce deletion is not approved",
            )
        ]
    if tool_name == "mcp__github__merge_pull_request":
        repo = tool_input.get("repo")
        if repo not in APPROVED_REPOS:
            return [
                Finding.fail(
                    "mcp_repository_scope",
                    severity="CRITICAL",
                    message=f"repository {repo!r} is outside the approved scope",
                )
            ]
    return []


def _hook(event: dict, policy, **kw):
    out = io.StringIO()
    result = run_pretooluse_hook(policy, stdin=io.StringIO(json.dumps(event)), stdout=out, **kw)
    return result, out.getvalue()


# --- the call boundary: MCP names hit the same policy seam ------------------------------


def test_a_destructive_mcp_call_is_denied():
    decision, reason = decide("mcp__salesforce__delete_records", {"object": "Contact"}, mcp_policy)
    assert decision == "deny"
    assert "mcp__salesforce__delete_records" in reason  # the refusal names the MCP tool


def test_an_out_of_scope_mcp_call_is_denied_and_an_in_scope_one_defers():
    deny, reason = decide("mcp__github__merge_pull_request", {"repo": "attacker/repo"}, mcp_policy)
    assert deny == "deny" and "attacker/repo" in reason
    defer, _ = decide("mcp__github__merge_pull_request", {"repo": "philpaz/recusal"}, mcp_policy)
    assert defer == "defer"


def test_a_clean_mcp_call_defers_it_never_auto_allows():
    # PASS defers to Claude Code's own permission flow; the gate only ever adds refusals.
    decision, _ = decide("mcp__github__create_issue", {"repo": "philpaz/recusal"}, mcp_policy)
    assert decision == "defer"


# --- end to end: a real PreToolUse event carrying an MCP tool name ----------------------


def test_an_mcp_event_flows_through_the_hook_like_any_native_tool():
    event = {"tool_name": "mcp__salesforce__delete_records", "tool_input": {"object": "Contact"}}
    result, emitted = _hook(event, mcp_policy)
    assert result is not None
    hso = result["hookSpecificOutput"]
    assert hso["hookEventName"] == "PreToolUse"
    assert hso["permissionDecision"] == "deny"
    assert "mcp__salesforce__delete_records" in hso["permissionDecisionReason"]
    assert json.loads(emitted) == result  # the wire output is the decision, verbatim


def test_a_deferring_mcp_event_emits_nothing():
    event = {"tool_name": "mcp__github__create_issue", "tool_input": {"title": "x"}}
    result, emitted = _hook(event, mcp_policy)
    assert result is None and emitted == ""  # Claude Code's normal permission flow runs


# --- allowlist mode: default-deny already covers MCP ------------------------------------


def test_allowlist_mode_refuses_mcp_tools_unless_affirmatively_named():
    strict = allowlist_policy()
    decision, reason = decide("mcp__github__create_issue", {"repo": "philpaz/recusal"}, strict)
    assert decision == "deny" and "not on the allowlist" in reason


def test_an_allow_predicate_affirmatively_vets_a_named_mcp_tool():
    vetted = allowlist_policy(
        allow={"mcp__github__create_issue": lambda i: i.get("repo") in APPROVED_REPOS}
    )
    ok, _ = decide("mcp__github__create_issue", {"repo": "philpaz/recusal"}, vetted)
    assert ok == "defer"  # vetted still defers; the predicate never force-allows
    bad, _ = decide("mcp__github__create_issue", {"repo": "attacker/repo"}, vetted)
    assert bad == "deny"


# --- robustness: the gate's failure posture holds for MCP events too --------------------


def test_a_buggy_policy_fails_closed_on_an_mcp_event():
    def buggy(tool_name, tool_input):
        if tool_name.startswith("mcp__"):
            raise RuntimeError("policy bug")
        return []

    decision, reason = decide("mcp__github__create_issue", {}, buggy)
    assert decision == "deny" and "failed closed" in reason
