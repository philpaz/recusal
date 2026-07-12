"""``manifest_policy``: the pinned MCP manifest enforced inside the call-time gate.

The bridge's contract: a pinned ``mcp__server__tool`` call defers (never auto-allows), an
unpinned one refuses, a missing or corrupt manifest fails CLOSED for MCP calls (no pin,
no MCP), non-MCP tools pass through untouched, and argument-level policies compose on
top of the pin.
"""

import io
import json

from recusal import Finding
from recusal.claude_code import decide, run_pretooluse_hook
from recusal.mcp import build_manifest, manifest_policy, manifest_to_text


def _tool(name="create_issue", description="Create a GitHub issue."):
    return {"name": name, "description": description, "inputSchema": {"type": "object"}}


def _pin(tmp_path):
    path = tmp_path / "mcp-manifest.json"
    path.write_text(manifest_to_text(build_manifest({"github": [_tool()]})), encoding="utf-8")
    return str(path)


def test_a_pinned_mcp_call_defers_never_auto_allows(tmp_path):
    policy = manifest_policy(_pin(tmp_path))
    assert decide("mcp__github__create_issue", {"title": "x"}, policy)[0] == "defer"


def test_an_unpinned_tool_and_an_unpinned_server_are_refused(tmp_path):
    policy = manifest_policy(_pin(tmp_path))
    for tool in ("mcp__github__delete_repo", "mcp__pastebin__upload"):
        decision, reason = decide(tool, {}, policy)
        assert decision == "deny" and tool in reason


def test_a_missing_manifest_fails_closed_for_mcp_calls(tmp_path):
    policy = manifest_policy(str(tmp_path / "nope.json"))
    decision, reason = decide("mcp__github__create_issue", {}, policy)
    assert decision == "deny" and "no pin, no MCP" in reason


def test_a_corrupt_manifest_fails_closed(tmp_path):
    path = tmp_path / "mcp-manifest.json"
    path.write_text("{ not json", encoding="utf-8")
    assert decide("mcp__github__create_issue", {}, manifest_policy(str(path)))[0] == "deny"


def test_non_mcp_tools_are_not_the_bridges_business(tmp_path):
    # No inner policy: native tools defer, even with no manifest on disk at all.
    policy = manifest_policy(str(tmp_path / "nope.json"))
    assert decide("Bash", {"command": "rm -rf /"}, policy)[0] == "defer"


def test_a_server_name_containing_double_underscore_matches_by_full_name(tmp_path):
    # Membership is by the exact runtime name mcp__{server}__{tool}, reconstructed from the
    # pin, so a server whose name contains "__" is neither mis-denied nor mis-attributed to
    # a different pinned tool. (Regression for the name-split ambiguity.)
    path = tmp_path / "mcp-manifest.json"
    path.write_text(
        manifest_to_text(build_manifest({"github__issues": [_tool("create")]})),
        encoding="utf-8",
    )
    policy = manifest_policy(str(path))
    # the real runtime name for that pinned tool defers...
    assert decide("mcp__github__issues__create", {}, policy)[0] == "defer"
    # ...while a plausible mis-split (server "github", tool "issues__create") is NOT pinned
    # under that reading either, and there is no such tool -> still the same call, defers once
    # (there is exactly one interpretation that matches the pin, and it is the right one).
    assert decide("mcp__github__issues__delete", {}, policy)[0] == "deny"


def test_argument_level_rules_compose_on_top_of_the_pin(tmp_path):
    def repo_scope(tool_name, tool_input):
        if tool_name == "mcp__github__create_issue" and tool_input.get("repo") != "philpaz/recusal":
            return [
                Finding.fail(
                    "mcp_repository_scope",
                    severity="CRITICAL",
                    message="repository is outside the approved scope",
                )
            ]
        return []

    policy = manifest_policy(_pin(tmp_path), policy=repo_scope)
    # pinned AND in scope -> defer; pinned but out of scope -> the inner rule still refuses
    assert decide("mcp__github__create_issue", {"repo": "philpaz/recusal"}, policy)[0] == "defer"
    assert decide("mcp__github__create_issue", {"repo": "attacker/repo"}, policy)[0] == "deny"

    # and the inner policy also still sees native tools
    def no_bash(tool_name, tool_input):
        if tool_name == "Bash":
            return [Finding.fail("no_bash", severity="CRITICAL", message="no shell")]
        return []

    assert decide("Bash", {}, manifest_policy(_pin(tmp_path), policy=no_bash))[0] == "deny"


def test_end_to_end_through_the_hook(tmp_path):
    policy = manifest_policy(_pin(tmp_path))
    event = {"tool_name": "mcp__pastebin__upload", "tool_input": {"content": "secrets"}}
    out = io.StringIO()
    result = run_pretooluse_hook(policy, stdin=io.StringIO(json.dumps(event)), stdout=out)
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "mcp__pastebin__upload" in result["hookSpecificOutput"]["permissionDecisionReason"]


# --- the mtime/size-keyed manifest cache: cheap per call, re-pins picked up live ----------


def test_a_repin_is_picked_up_live_without_a_new_policy(tmp_path):
    path = tmp_path / "mcp-manifest.json"
    path.write_text(manifest_to_text(build_manifest({"github": [_tool()]})), encoding="utf-8")
    policy = manifest_policy(str(path))
    assert decide("mcp__github__create_issue", {}, policy)[0] == "defer"
    assert decide("mcp__github__close_issue", {}, policy)[0] == "deny"

    # re-pin with a different catalog; the SAME policy object must see it
    path.write_text(
        manifest_to_text(build_manifest({"github": [_tool("close_issue")]})), encoding="utf-8"
    )
    assert decide("mcp__github__close_issue", {}, policy)[0] == "defer"
    assert decide("mcp__github__create_issue", {}, policy)[0] == "deny"


def test_an_unchanged_manifest_is_parsed_once_across_calls(tmp_path, monkeypatch):
    # The bytes are read every call (no staleness); the parse+validate work happens only
    # when the content digest changes.
    import recusal.mcp as mcp_mod

    path = tmp_path / "mcp-manifest.json"
    path.write_text(manifest_to_text(build_manifest({"github": [_tool()]})), encoding="utf-8")
    calls = {"n": 0}
    real = mcp_mod._validate_manifest

    def _counting(data):
        calls["n"] += 1
        return real(data)

    monkeypatch.setattr(mcp_mod, "_validate_manifest", _counting)
    policy = manifest_policy(str(path))
    for _ in range(5):
        assert decide("mcp__github__create_issue", {}, policy)[0] == "defer"
    assert calls["n"] == 1


def test_a_manifest_deleted_after_caching_fails_closed(tmp_path):
    # The cache must never serve a pin past its file: no file, no pin, no MCP.
    path = tmp_path / "mcp-manifest.json"
    path.write_text(manifest_to_text(build_manifest({"github": [_tool()]})), encoding="utf-8")
    policy = manifest_policy(str(path))
    assert decide("mcp__github__create_issue", {}, policy)[0] == "defer"
    path.unlink()
    decision, reason = decide("mcp__github__create_issue", {}, policy)
    assert decision == "deny" and "no pin, no MCP" in reason


def test_a_manifest_corrupted_after_caching_fails_closed(tmp_path):
    # Changed-but-unreadable must refuse, not fall back to the stale cached pin.
    path = tmp_path / "mcp-manifest.json"
    path.write_text(manifest_to_text(build_manifest({"github": [_tool()]})), encoding="utf-8")
    policy = manifest_policy(str(path))
    assert decide("mcp__github__create_issue", {}, policy)[0] == "defer"
    path.write_text("{ not json", encoding="utf-8")
    assert decide("mcp__github__create_issue", {}, policy)[0] == "deny"


def test_a_revocation_with_preserved_mtime_and_size_is_seen(tmp_path):
    # An (mtime, size) signature would miss this exact case: same byte length, same
    # timestamp, different content - a REVOCATION landing via deployment tooling. The
    # cache is keyed on the content digest, so stale authorization cannot be served.
    import os

    path = tmp_path / "mcp-manifest.json"
    original = manifest_to_text(build_manifest({"github": [_tool("create_issue")]}))
    path.write_text(original, encoding="utf-8")
    stat = os.stat(path)
    policy = manifest_policy(str(path))
    assert decide("mcp__github__create_issue", {}, policy)[0] == "defer"

    # revoke create_issue in favor of an equal-length name, so the file size is identical
    # (both writes get the same newline translation, so equal text = equal st_size)
    replacement = manifest_to_text(build_manifest({"github": [_tool("delete_issue")]}))
    assert len(replacement) == len(original)  # the signature-busting premise
    path.write_text(replacement, encoding="utf-8")
    assert os.path.getsize(path) == stat.st_size
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # preserve the timestamp

    assert decide("mcp__github__create_issue", {}, policy)[0] == "deny"  # revoked
    assert decide("mcp__github__delete_issue", {}, policy)[0] == "defer"
