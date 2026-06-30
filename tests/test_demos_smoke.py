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
