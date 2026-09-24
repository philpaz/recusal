"""Literal destination boundaries of the cookbook example; no requests are sent."""

import runpy
from pathlib import Path

import pytest

from recusal import compute_verdict

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "egress_allowlist.py"
make_policy = runpy.run_path(str(EXAMPLE))["make_egress_allowlist"]


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("send_email", {"to": "person@example.com"}),
        ("send_email", {"to": "person@sub.example.com"}),
        ("http_post", {"url": "https://EXAMPLE.COM/path?q=1"}),
        ("webhook", {"url": "http://api.example.com:8080/path"}),
    ],
)
def test_allowed_destinations_defer(tool, arguments):
    assert make_policy({"example.com"})(tool, arguments) == []


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("send_email", {"to": "person@evil-example.com"}),
        ("send_email", {"to": "attacker@evil.com"}),
        ("send_email", {"to": "a@evil.com,b@example.com"}),
        ("send_email", {"to": "@example.com"}),
        ("webhook", {"url": "https://example.com.evil.com/"}),
        ("webhook", {"url": "https://example.com@evil.com/"}),
        ("http_post", {"url": "https://evil-example.com/"}),
        ("http_post", {"url": "https://[invalid"}),
        ("http_post", {"url": "file://example.com/file"}),
        ("http_post", {"to": "safe@example.com", "url": "https://evil.com/"}),
        ("http_post", {"url": None}),
        ("send_email", {}),
    ],
)
def test_disallowed_or_missing_destinations_refuse(tool, arguments):
    findings = make_policy({"example.com"})(tool, arguments)
    assert compute_verdict(findings).refused
    assert findings[0].check == "egress_allowlist"


def test_runtime_built_destination_is_not_inspected():
    # No shell is launched; checking the literal tool envelope is the whole boundary.
    assert make_policy({"example.com"})("Bash", {"command": "python send.py"}) == []


def test_runnable_demo_states_its_boundary(capsys):
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    output = capsys.readouterr().out
    assert "lookalike host: DENY" in output
    assert "allowed email: DEFER" in output
    assert "NOT INSPECTED" in output
