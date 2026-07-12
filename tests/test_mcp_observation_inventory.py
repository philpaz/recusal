"""Review 9 (0.5.6): whole-server observation completeness, strict container
validation, manifest-before-business-policy ordering, and reserved Claude MCP names.

The trust gaps these close: (1) a pinned server absent from EVERY observation
component produced only a warning, so a partial multi-server observation verified
clean while the manifest kept authorizing that server's pinned runtime names; (2)
`unverifiable` accepted any iterable (a string iterated to characters) and source
values were not structurally validated; (3) `manifest_policy` invoked the wrapped
business policy before establishing manifest membership, so an unapproved capability
triggered downstream policy work; (4) the parser represented (and would launch)
config entries using names Claude Code reserves for built-in servers and skips.
"""

import io
import json

import pytest

from recusal import compute_verdict
from recusal.__main__ import mcp_pin_command, mcp_verify_command
from recusal.mcp import (
    McpObservation,
    build_manifest,
    diff_observation,
    manifest_policy,
    manifest_to_text,
)
from recusal.mcp_fetch import _CLAUDE_RESERVED_MCP_NAMES, servers_from_claude_config

TWO_SERVER_CATALOG = {
    "github": [{"name": "search", "description": "search code"}],
    "banking": [{"name": "get_account", "description": "read one account"}],
}


def _two_server_pin():
    return build_manifest(
        TWO_SERVER_CATALOG,
        instructions={"github": "gh instructions", "banking": "bank instructions"},
    )


def _github_only_observation(**kwargs):
    return McpObservation(
        catalog={"github": TWO_SERVER_CATALOG["github"]},
        sources={"github": {"transport": "external"}},
        instructions={"github": {"observed": True, "text": "gh instructions"}},
        **kwargs,
    )


def _failed_checks(findings):
    return {f.check for f in findings if not f.passed}


# --- P0-1: a wholly omitted pinned server refuses ---------------------------------------


def test_a_wholly_omitted_pinned_server_is_a_critical_refusal():
    # github fully verifies; banking appears NOWHERE - previously a passing WARNING,
    # which let a partial observation verify clean while the manifest kept
    # authorizing banking's pinned runtime names
    findings = diff_observation(_two_server_pin(), _github_only_observation())
    assert "mcp_server_unobserved" in _failed_checks(findings)
    assert not compute_verdict(findings).passed


def test_an_acknowledged_removal_is_a_recorded_nonblocking_transition():
    findings = diff_observation(_two_server_pin(), _github_only_observation(removed=("banking",)))
    verdict = compute_verdict(findings)
    assert verdict.passed, verdict.reasons()
    removed = [f for f in findings if f.check == "mcp_server_removed"]
    assert removed and removed[0].severity.value == "WARNING"


def test_an_omitted_server_marked_unverifiable_refuses():
    findings = diff_observation(
        _two_server_pin(), _github_only_observation(unverifiable=("banking",))
    )
    assert "mcp_pinned_server_unverifiable" in _failed_checks(findings)
    assert not compute_verdict(findings).passed


def test_a_removed_name_the_manifest_does_not_pin_is_rejected():
    with pytest.raises(ValueError, match="does not pin"):
        diff_observation(_two_server_pin(), _github_only_observation(removed=("ghost",)))


def test_a_removed_name_that_is_also_represented_is_contradictory():
    with pytest.raises(ValueError, match="contradicts"):
        diff_observation(_two_server_pin(), _github_only_observation(removed=("github",)))


def test_a_repinned_manifest_without_the_server_verifies_clean():
    repinned = build_manifest(
        {"github": TWO_SERVER_CATALOG["github"]},
        instructions={"github": "gh instructions"},
    )
    verdict = compute_verdict(diff_observation(repinned, _github_only_observation()))
    assert verdict.passed


def test_the_pinned_runtime_name_stays_authorized_which_is_why_omission_must_refuse(tmp_path):
    # the risk the refusal closes: while the manifest still pins banking, its runtime
    # tool name remains authorized at call time - so a verification that silently
    # skips banking is not evidence that banking is safe
    manifest = tmp_path / "m.json"
    manifest.write_text(manifest_to_text(_two_server_pin()), encoding="utf-8")
    policy = manifest_policy(str(manifest))
    assert policy("mcp__banking__get_account", {}) == []  # still authorized
    findings = diff_observation(_two_server_pin(), _github_only_observation())
    assert not compute_verdict(findings).passed  # so partial verification must refuse


def test_cli_from_only_partial_dump_refuses_and_removed_acknowledges(tmp_path):
    manifest = tmp_path / "m.json"
    dump = tmp_path / "obs.json"
    rich = {
        name: {"instructions": f"{name} instructions"[:60], "tools": tools}
        for name, tools in TWO_SERVER_CATALOG.items()
    }
    rich["github"]["instructions"] = "gh instructions"
    rich["banking"]["instructions"] = "bank instructions"
    dump.write_text(json.dumps(rich), encoding="utf-8")
    assert mcp_pin_command(str(manifest), from_file=str(dump), stdout=io.StringIO()) == 0

    partial = {"github": rich["github"]}
    dump.write_text(json.dumps(partial), encoding="utf-8")
    out = io.StringIO()
    rc = mcp_verify_command(str(manifest), from_file=str(dump), stdout=out)
    assert rc == 2 and "mcp_server_unobserved" in out.getvalue()

    out = io.StringIO()
    rc = mcp_verify_command(str(manifest), from_file=str(dump), removed=["banking"], stdout=out)
    assert rc == 0, out.getvalue()
    assert "acknowledged as deliberately removed" in out.getvalue()


# --- P1-1: strict containers and source values -------------------------------------------


@pytest.mark.parametrize(
    "bad",
    ["srv", {"srv": True}, None, {"srv"}, 7],
    ids=["string", "dict", "none", "set", "int"],
)
def test_malformed_unverifiable_containers_raise_valueerror(bad):
    pinned = _two_server_pin()
    with pytest.raises(ValueError, match="unverifiable"):
        diff_observation(pinned, McpObservation(catalog={}, unverifiable=bad))


@pytest.mark.parametrize(
    "bad",
    ["srv", None, {"srv": True}],
    ids=["string", "none", "dict"],
)
def test_malformed_removed_containers_raise_valueerror(bad):
    pinned = _two_server_pin()
    with pytest.raises(ValueError, match="removed"):
        diff_observation(pinned, McpObservation(catalog={}, removed=bad))


@pytest.mark.parametrize(
    "bad_source",
    [None, "stdio", ["stdio"], {"transport": "carrier-pigeon"}, {"transport": "stdio"}],
    ids=["none", "string", "list", "bad-transport", "stdio-no-command"],
)
def test_malformed_source_values_raise_valueerror_even_for_unpinned_servers(bad_source):
    pinned = _two_server_pin()
    obs = McpObservation(catalog={}, sources={"ghost": bad_source})
    with pytest.raises(ValueError, match="source observation for server 'ghost'"):
        diff_observation(pinned, obs)


def test_contradictory_source_fields_raise_valueerror():
    pinned = _two_server_pin()
    obs = McpObservation(
        catalog={},
        sources={"ghost": {"transport": "external", "command": "py"}},
    )
    with pytest.raises(ValueError, match="source observation for server 'ghost'"):
        diff_observation(pinned, obs)


# --- P1-2: manifest membership before the wrapped business policy -------------------------


def _counting_policy():
    calls = []

    def inner(tool_name, tool_input):
        calls.append(tool_name)
        return []

    return inner, calls


def test_wrapped_policy_ordering(tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text(manifest_to_text(_two_server_pin()), encoding="utf-8")
    inner, calls = _counting_policy()
    policy = manifest_policy(str(manifest), policy=inner)

    assert policy("Bash", {"command": "echo hi"}) == []
    assert calls == ["Bash"]  # non-MCP: wrapped policy runs

    assert policy("mcp__github__search", {}) == []
    assert calls == ["Bash", "mcp__github__search"]  # pinned MCP: runs after membership

    findings = policy("mcp__evil__exfil", {})
    assert any(f.check == "mcp_not_pinned" for f in findings)
    assert calls == ["Bash", "mcp__github__search"], (
        "an unapproved capability triggered downstream business-policy work"
    )


def test_wrapped_policy_not_called_on_missing_or_corrupt_manifest(tmp_path):
    inner, calls = _counting_policy()
    policy = manifest_policy(str(tmp_path / "absent.json"), policy=inner)
    findings = policy("mcp__github__search", {})
    assert any(f.check == "mcp_manifest_unavailable" for f in findings)
    assert calls == []

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    inner2, calls2 = _counting_policy()
    policy2 = manifest_policy(str(corrupt), policy=inner2)
    findings = policy2("mcp__github__search", {})
    assert any(f.check == "mcp_manifest_unavailable" for f in findings)
    assert calls2 == []


def test_inner_argument_policy_still_refuses_a_pinned_call(tmp_path):
    from recusal import Finding

    manifest = tmp_path / "m.json"
    manifest.write_text(manifest_to_text(_two_server_pin()), encoding="utf-8")

    def strict_inner(tool_name, tool_input):
        return [Finding.fail("arg_rule", severity="CRITICAL", message="wrong subject")]

    policy = manifest_policy(str(manifest), policy=strict_inner)
    findings = policy("mcp__github__search", {"q": "x"})
    assert not compute_verdict(findings).passed


# --- P1-3: reserved Claude MCP server names ------------------------------------------------


@pytest.mark.parametrize("reserved", sorted(_CLAUDE_RESERVED_MCP_NAMES))
def test_every_reserved_name_refuses_before_any_launch(tmp_path, reserved):
    marker = tmp_path / "EXECUTED.marker"
    config = tmp_path / ".mcp.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    reserved: {
                        "command": "py",
                        "args": ["-c", f"open(r'{marker}', 'w').write('ran')"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reserved by Claude Code"):
        servers_from_claude_config(str(config))
    assert not marker.exists(), "a reserved-name entry's command executed"

    out = io.StringIO()
    rc = mcp_pin_command(
        str(tmp_path / "m.json"),
        claude_config=str(config),
        approve_server_launch=True,
        stdout=out,
    )
    assert rc == 2 and not marker.exists()


def test_a_reserved_remote_entry_also_refuses(tmp_path):
    config = tmp_path / ".mcp.json"
    config.write_text(
        json.dumps(
            {"mcpServers": {"workspace": {"type": "http", "url": "https://example.test/mcp"}}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reserved by Claude Code"):
        servers_from_claude_config(str(config))


def test_ordinary_names_continue_to_parse(tmp_path):
    config = tmp_path / ".mcp.json"
    config.write_text(
        json.dumps(
            {"mcpServers": {"my-workspace": {"type": "http", "url": "https://example.test/mcp"}}}
        ),
        encoding="utf-8",
    )
    _, remote = servers_from_claude_config(str(config))
    assert "my-workspace" in remote
