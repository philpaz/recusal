"""``allowlist_policy(allow=...)``: an explicit predicate is the whole decision for its tool.

Found by an outside review: the read-only defaults (Read/Grep/Glob) were checked before
``allow``, so a predicate meant to narrow a read (``Read`` of a credentials file) never
ran and the call deferred. The docstring always said the predicate overrides the
built-in vetting; these tests hold the code to it, and pin that nothing else moved.
"""

import io
import json

import pytest

from recusal.claude_code import allowlist_policy, decide, run_pretooluse_hook

READ_ONLY = {
    "Read": {"file_path": "credentials.json"},
    "Grep": {"pattern": "password", "path": "."},
    "Glob": {"pattern": "**/*.pem"},
}


@pytest.mark.parametrize("tool", sorted(READ_ONLY))
def test_a_refusing_predicate_refuses_a_default_read_only_tool(tool):
    ran = []

    def never(args):
        ran.append(args)
        return False

    decision, reason = decide(tool, READ_ONLY[tool], allowlist_policy(allow={tool: never}))
    assert decision == "deny"
    assert "predicate refused it" in reason
    assert ran == [READ_ONLY[tool]]


@pytest.mark.parametrize("tool", sorted(READ_ONLY))
def test_a_passing_predicate_defers_a_default_read_only_tool(tool):
    policy = allowlist_policy(allow={tool: lambda args: True})
    assert decide(tool, READ_ONLY[tool], policy)[0] == "defer"


def test_the_predicate_sees_the_arguments_and_narrows_reads():
    def no_secrets(args):
        return not str(args.get("file_path", "")).endswith((".env", "credentials.json"))

    policy = allowlist_policy(allow={"Read": no_secrets})
    assert decide("Read", {"file_path": "src/app.py"}, policy)[0] == "defer"
    assert decide("Read", {"file_path": "credentials.json"}, policy)[0] == "deny"
    assert decide("Read", {"file_path": "deploy/.env"}, policy)[0] == "deny"
    # the other read-only defaults are untouched by a predicate on Read
    assert decide("Grep", {"pattern": "x"}, policy)[0] == "defer"


@pytest.mark.parametrize("tool", sorted(READ_ONLY))
def test_a_predicate_that_raises_fails_closed(tool):
    def boom(args):
        raise RuntimeError("predicate exploded")

    decision, reason = decide(tool, READ_ONLY[tool], allowlist_policy(allow={tool: boom}))
    assert decision == "deny"
    assert "policy error" in reason


@pytest.mark.parametrize("tool", sorted(READ_ONLY))
def test_without_a_predicate_the_read_only_defaults_still_defer(tool):
    assert decide(tool, READ_ONLY[tool], allowlist_policy())[0] == "defer"


def test_other_tools_are_unchanged():
    policy = allowlist_policy(allow={"Read": lambda args: False})
    assert decide("Bash", {"command": "ls"}, policy)[0] == "defer"
    assert decide("Bash", {"command": "python x.py"}, policy)[0] == "deny"
    assert decide("Write", {"file_path": "/tmp/x"}, policy)[0] == "deny"
    assert decide("WebFetch", {"url": "https://x"}, policy)[0] == "deny"


def test_the_hook_applies_the_predicate_too():
    out = io.StringIO()
    event = {"hook_event_name": "PreToolUse", "tool_name": "Read", "tool_input": READ_ONLY["Read"]}
    result = run_pretooluse_hook(
        allowlist_policy(allow={"Read": lambda args: False}),
        stdin=io.StringIO(json.dumps(event)),
        stdout=out,
    )
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
