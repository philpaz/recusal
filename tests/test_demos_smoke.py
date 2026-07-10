"""Smoke tests, the offline demos actually run and produce the expected verdicts."""

import contextlib
import io
import os
import runpy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_demo(rel_path: str) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_path(os.path.join(REPO, rel_path), run_name="__main__")
    return buf.getvalue()


def test_claude_refusal_demo_refuses_then_allows():
    out = _run_demo("examples/claude_refusal.py")
    assert "REFUSED" in out
    assert "ALLOWED" in out
    assert "C-9988" in out  # the wrong-subject target appears in the refusal


def test_gallery_demo_shows_all_tiers():
    out = _run_demo("examples/gallery.py")
    assert "REFUSE" in out
    assert "BLOCK (retry)" in out
    assert "ALLOW" in out


def test_quickstart_demo_runs():
    out = _run_demo("examples/quickstart.py")
    assert "FAIL" in out
    assert "release_ready" in out


def test_framework_neutral_agent_loop_gates_tool_calls():
    out = _run_demo("examples/agent_loop.py")
    assert "REFUSE" in out  # destructive / unscoped actions blocked terminally
    assert "RETRY" in out  # recoverable (allowlist) action blocked for retry
    assert "ALLOW" in out  # compliant actions proceed
    assert "no Claude, no SDK" in out  # the claim this demo exists to prove


def test_injection_quarantine_demo_quarantines_poisoned_output():
    out = _run_demo("examples/injection_quarantine.py")
    # Clean observations pass; poisoned ones are refused and routed to quarantine (not
    # ask-human, i.e. the screen and the router share one vocabulary).
    assert "safe, use as context" in out
    assert "quarantine" in out
    assert "ask-human" not in out
    assert "2 of 4 observations quarantined" in out


def test_mcp_governance_demo_governs_mcp_calls():
    out = _run_demo("examples/mcp_governance.py")
    # The claim the demo exists to prove: MCP calls hit the same gate as native tools.
    assert "mcp__<server>__<tool>" in out
    body = out.split("Allowlist mode", 1)[0]
    for line in body.splitlines():
        if any(
            k in line
            for k in ("Salesforce delete", "foreign repo", "unknown server", "outside workspace")
        ):
            assert "DENY" in line, (
                line
            )  # destructive / out-of-scope / unapproved-server MCP calls refuse
        if any(
            k in line for k in ("create an issue", "approved repo", "inside workspace", "not MCP")
        ):
            assert "DEFER" in line, (
                line
            )  # clean calls defer to Claude Code's own flow, never auto-allow
    tail = out.split("Allowlist mode", 1)[1]
    assert "unlisted MCP tool" in tail  # default-deny already covers MCP
    for line in tail.splitlines():
        if "unlisted MCP tool" in line or "predicate refuses" in line:
            assert "DENY" in line, line
        if "predicate passes" in line:
            assert "DEFER" in line, line
    assert "CALL-TIME" in out  # the demo states its boundary honestly


def test_mcp_full_stack_demo_covers_all_three_boundaries():
    out = _run_demo("examples/mcp_full_stack.py")
    for line in out.splitlines():
        if any(
            k in line for k in ("out of scope", "NOT pinned", "unpinned server", "manifest missing")
        ):
            assert "DENY" in line, line  # discovery + invocation refusals
        if any(k in line for k in ("no rule -> defer", "in scope -> defer", "not an MCP call")):
            assert "DEFER" in line, line
    assert "QUARANTINE (do not trust)" in out  # the response boundary
    assert "trust as context" in out  # a clean result is allowed through


def test_mcp_rugpull_demo_pins_then_refuses_drift():
    out = _run_demo("examples/mcp_manifest_rugpull.py")
    assert "same catalog re-observed" in out and "PASS" in out
    for line in out.splitlines():
        if "rewritten after approval" in line or "unreviewed tool appeared" in line:
            assert line.rstrip().endswith("FAIL"), line
    calltime = out.split("enforced at call time", 1)[1]
    for line in calltime.splitlines():
        if "create_issue" in line and "missing" not in line:
            assert "DEFER" in line, line  # pinned -> defers to Claude Code's own flow
        if "delete_repository" in line or "manifest missing" in line:
            assert "DENY" in line, line  # unpinned or unpinnable -> refused
    assert "rug-pull" in out  # the refusal names the vector
    assert "no pin, no MCP" in out  # a missing manifest fails closed at call time


def test_allowlist_gate_demo_clears_the_denylist_ceiling():
    out = _run_demo("examples/allowlist_gate.py")
    # The teaching moment: a de-obfuscating deny-list still DEFERS the runtime-constructed
    # names, while the default-deny allowlist DENIES every unvetted call.
    assert "hex-built name" in out
    assert "DEFER" in out  # the deny-list column lets a constructed name through
    assert "default-deny" in out
    # The allowlist column must refuse all five attacks (never a bare DEFER before "vetted").
    body = out.split("run pytest")[0]
    for line in body.splitlines():
        if any(k in line for k in ("rm -rf", "built name", "base64")):
            assert line.rstrip().endswith("DENY"), line  # allowlist (last column) refuses
