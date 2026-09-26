"""Unknown tool identity never passes an allowlist."""

import pytest

from examples.approved_tool_policy import policy
from recusal import compute_verdict


@pytest.mark.parametrize("name", ["Read", "Grep", "Glob", "Bash", "update_record"])
def test_approved_name_defers(name):
    assert policy(name, {}) == []


@pytest.mark.parametrize(
    "name",
    [
        "Write",
        "mcp__evil__exfiltrate",
        "mcp__trusted__Read",
        "",
        None,
        0,
        False,
        [],
        {},
        "read",
        " Read",
        "Read ",
        "Read\x00",
    ],
)
def test_unapproved_identity_refuses(name):
    assert compute_verdict(policy(name, {})).refused


def test_missing_name_from_event_refuses():
    event = {"tool_input": {}}
    assert compute_verdict(policy(event.get("tool_name"), event["tool_input"])).refused
