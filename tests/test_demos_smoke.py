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
