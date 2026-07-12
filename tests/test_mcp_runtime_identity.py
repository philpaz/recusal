"""Manifest v6 runtime identity (review 11/12 carry-forward, shipped as 0.5.9): raw
MCP declaration identity and Claude callable identity are modeled separately, with an
EXPLICIT per-server runtime naming mode.

The trust gap this closes: Claude documents that any character outside A-Z a-z 0-9 _ -
is replaced with "_" in plugin MCP callable names, while the MCP specification permits
more in raw tool names (a dotted `admin.tools.list` is spec-valid). v5 reconstructed
PreToolUse names from raw declaration names, so a spec-valid plugin tool was refused
under Claude's normalized spelling (false denial), and two raw spellings could collide
into one callable identity without detection. v6: raw identity for discovery and drift
verification, callable identity for PreToolUse membership, collisions refuse the pin,
and the mode is explicit in the manifest - never inferred from a key's spelling.
"""

import hashlib
import io
import json

import pytest

from recusal import compute_verdict
from recusal.__main__ import mcp_pin_command, mcp_verify_command
from recusal.claude_code import _control_identity
from recusal.mcp import (
    MANIFEST_VERSION,
    McpObservation,
    build_manifest,
    diff_observation,
    load_manifest,
    manifest_policy,
    manifest_to_text,
    plugin_callable_name,
)

PLUGIN_SERVER = "plugin_my-plugin_database-tools"
DOTTED = {"name": "admin.tools.list", "description": "list admin tools"}
SAFE = {"name": "query", "description": "run a query"}


def _plugin_pin(tools=(DOTTED, SAFE)):
    return build_manifest(
        {PLUGIN_SERVER: list(tools)},
        instructions={PLUGIN_SERVER: "approved"},
        runtime_modes={PLUGIN_SERVER: "claude_plugin"},
    )


def _write(tmp_path, manifest):
    path = tmp_path / "m.json"
    path.write_text(manifest_to_text(manifest), encoding="utf-8")
    return str(path)


def _failed_checks(findings):
    return {f.check for f in findings if not f.passed}


# --- the normalization rule itself -----------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("query", "query"),  # safe name unchanged
        ("admin.tools.list", "admin_tools_list"),  # documented dot replacement
        ("DATA_EXPORT_v2", "DATA_EXPORT_v2"),  # case and underscores preserved
        ("user-profile/update", "user-profile_update"),  # spec-valid slash replaced
        ("a b\tc", "a_b_c"),  # whitespace replaced
    ],
)
def test_plugin_callable_name_matches_the_documented_rule(raw, expected):
    assert plugin_callable_name(raw) == expected


# --- pin: both identities stored, collisions refused ----------------------------------------


def test_a_plugin_pin_stores_raw_and_callable_identity():
    manifest = _plugin_pin()
    entry = manifest["servers"][PLUGIN_SERVER]
    assert manifest["manifest_version"] == MANIFEST_VERSION == 6
    assert entry["runtime"] == {"mode": "claude_plugin"}
    pin = entry["tools"]["admin.tools.list"]  # raw declaration name IS the key
    assert pin["callable_name"] == "admin_tools_list"  # callable identity alongside
    assert pin["fingerprint"].startswith("sha256:")  # raw declaration fingerprint kept


def test_standard_mode_pins_carry_no_callable_identity():
    manifest = build_manifest({"srv": [DOTTED]})
    entry = manifest["servers"]["srv"]
    assert entry["runtime"] == {"mode": "standard_mcp"}
    assert "callable_name" not in entry["tools"]["admin.tools.list"]


def test_plugin_mode_is_never_inferred_from_the_key_prefix():
    # a server key SPELLED like a plugin segment, without the explicit mode, stays
    # standard: runtime semantics are declared, not guessed
    manifest = build_manifest({PLUGIN_SERVER: [DOTTED]})
    assert manifest["servers"][PLUGIN_SERVER]["runtime"] == {"mode": "standard_mcp"}


def test_colliding_raw_names_refuse_the_pin():
    with pytest.raises(ValueError, match="ambiguous callable identity"):
        build_manifest(
            {PLUGIN_SERVER: [{"name": "admin.tools.list"}, {"name": "admin_tools_list"}]},
            runtime_modes={PLUGIN_SERVER: "claude_plugin"},
        )


def test_an_unsafe_plugin_server_key_refuses_rather_than_rewriting():
    with pytest.raises(ValueError, match="callable-safe"):
        build_manifest(
            {"plugin.dotted.key": [SAFE]}, runtime_modes={"plugin.dotted.key": "claude_plugin"}
        )


def test_unknown_mode_and_unknown_server_refuse():
    with pytest.raises(ValueError, match="runtime mode"):
        build_manifest({"s": [SAFE]}, runtime_modes={"s": "plugin"})
    with pytest.raises(ValueError, match="not in the catalog"):
        build_manifest({"s": [SAFE]}, runtime_modes={"ghost": "claude_plugin"})


# --- call time: callable identity authorizes, raw spelling does not --------------------------


def test_pretooluse_membership_uses_the_callable_identity(tmp_path):
    policy = manifest_policy(_write(tmp_path, _plugin_pin()))
    assert policy(f"mcp__{PLUGIN_SERVER}__admin_tools_list", {}) == []  # Claude's spelling
    assert policy(f"mcp__{PLUGIN_SERVER}__query", {}) == []  # safe name unchanged
    dotted_call = policy(f"mcp__{PLUGIN_SERVER}__admin.tools.list", {})
    assert any(f.check == "mcp_not_pinned" for f in dotted_call), (
        "the raw dotted runtime spelling must not be treated as the Claude callable"
    )


def test_standard_mode_dotted_names_keep_raw_runtime_naming(tmp_path):
    policy = manifest_policy(_write(tmp_path, build_manifest({"srv": [DOTTED]})))
    assert policy("mcp__srv__admin.tools.list", {}) == []
    assert any(not f.passed for f in policy("mcp__srv__admin_tools_list", {}))


# --- discovery: raw identity verifies; drift and alias behavior ------------------------------


def test_discovery_verification_uses_the_raw_declaration(tmp_path):
    pinned = _plugin_pin()
    obs = McpObservation(
        catalog={PLUGIN_SERVER: [dict(DOTTED), dict(SAFE)]},
        sources={PLUGIN_SERVER: {"transport": "external"}},
        instructions={PLUGIN_SERVER: {"observed": True, "text": "approved"}},
    )
    assert compute_verdict(diff_observation(pinned, obs)).passed


def test_a_raw_name_swap_preserving_the_callable_is_caught_at_verify(tmp_path):
    # dynamic list_changed residual, v6 form: the swap keeps the SAME approved
    # callable at call time (point-in-time boundary), and the next verify refuses on
    # RAW identity - which is exactly why both identities are modeled
    pinned = _plugin_pin(tools=(DOTTED,))
    policy = manifest_policy(_write(tmp_path, pinned))
    assert policy(f"mcp__{PLUGIN_SERVER}__admin_tools_list", {}) == []  # before AND after swap
    swapped = McpObservation(
        catalog={PLUGIN_SERVER: [{"name": "admin_tools_list", "description": "list admin tools"}]},
        sources={PLUGIN_SERVER: {"transport": "external"}},
        instructions={PLUGIN_SERVER: {"observed": True, "text": "approved"}},
    )
    findings = diff_observation(pinned, swapped)
    assert "mcp_unpinned_tool" in _failed_checks(findings)
    assert not compute_verdict(findings).passed


# --- backward compatibility: v5 refuses with a migration message -----------------------------


def test_a_v5_manifest_is_refused_with_a_repin_instruction(tmp_path):
    manifest = _plugin_pin()
    manifest["manifest_version"] = 5
    for entry in manifest["servers"].values():
        del entry["runtime"]  # v5 shape had no runtime record
        for pin in entry["tools"].values():
            pin.pop("callable_name", None)
    path = tmp_path / "old.json"
    path.write_text(manifest_to_text(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime-identity modeling"):
        load_manifest(str(path))
    # and the call-time bridge fails closed on it, not open
    findings = manifest_policy(str(path))(f"mcp__{PLUGIN_SERVER}__query", {})
    assert any(f.check == "mcp_manifest_unavailable" for f in findings if not f.passed)


# --- loader strictness: runtime identity is validated, not decorative ------------------------


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda m: m["servers"][PLUGIN_SERVER].pop("runtime"), "canonical runtime record"),
        (
            lambda m: m["servers"][PLUGIN_SERVER].update({"runtime": {"mode": "plugin"}}),
            "canonical runtime record",
        ),
        (
            lambda m: m["servers"][PLUGIN_SERVER].update(
                {"runtime": {"mode": "claude_plugin", "extra": 1}}
            ),
            "canonical runtime record",
        ),
        (
            lambda m: m["servers"][PLUGIN_SERVER]["tools"]["admin.tools.list"].update(
                {"callable_name": "tampered_name"}
            ),
            "does not re-derive",
        ),
        (
            lambda m: m["servers"][PLUGIN_SERVER]["tools"]["admin.tools.list"].pop("callable_name"),
            "not canonical",
        ),
    ],
)
def test_noncanonical_runtime_identity_is_refused(tmp_path, mutate, match):
    manifest = _plugin_pin()
    mutate(manifest)
    path = tmp_path / "m.json"
    path.write_text(manifest_to_text(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_manifest(str(path))


def test_a_callable_name_on_a_standard_pin_is_refused(tmp_path):
    manifest = build_manifest({"srv": [SAFE]})
    manifest["servers"]["srv"]["tools"]["query"]["callable_name"] = "query"
    path = tmp_path / "m.json"
    path.write_text(manifest_to_text(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical"):
        load_manifest(str(path))


# --- provenance: runtime identity is inside the digest the audit records ---------------------


def test_runtime_identity_changes_the_manifest_digest_and_audit_provenance(tmp_path):
    catalog = {"safe_server": [SAFE]}
    standard = manifest_to_text(build_manifest(catalog))
    plugin = manifest_to_text(
        build_manifest(catalog, runtime_modes={"safe_server": "claude_plugin"})
    )
    assert standard != plugin  # naming mode is part of the artifact bytes
    path = tmp_path / "m.json"
    path.write_text(plugin, encoding="utf-8")
    expected = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    policy = manifest_policy(str(path))
    assert policy("mcp__safe_server__query", {}) == []
    identity = _control_identity(policy, None)
    assert identity["manifest_sha256"] == expected  # digest covers runtime identity


# --- CLI: the explicit flag end to end --------------------------------------------------------


def test_cli_claude_plugin_flag_pins_and_authorizes_the_callable(tmp_path):
    dump = tmp_path / "obs.json"
    dump.write_text(
        json.dumps({PLUGIN_SERVER: {"instructions": "approved", "tools": [DOTTED, SAFE]}}),
        encoding="utf-8",
    )
    manifest = tmp_path / "m.json"
    rc = mcp_pin_command(
        str(manifest),
        from_file=str(dump),
        claude_plugin=[PLUGIN_SERVER],
        stdout=io.StringIO(),
    )
    assert rc == 0
    entry = load_manifest(str(manifest))["servers"][PLUGIN_SERVER]
    assert entry["runtime"] == {"mode": "claude_plugin"}
    assert entry["tools"]["admin.tools.list"]["callable_name"] == "admin_tools_list"

    # verify needs no flag: the mode lives in the manifest
    out = io.StringIO()
    assert mcp_verify_command(str(manifest), from_file=str(dump), stdout=out) == 0

    policy = manifest_policy(str(manifest))
    assert policy(f"mcp__{PLUGIN_SERVER}__admin_tools_list", {}) == []


def test_cli_collision_refuses_the_pin(tmp_path):
    dump = tmp_path / "obs.json"
    dump.write_text(
        json.dumps(
            {
                PLUGIN_SERVER: {
                    "instructions": None,
                    "tools": [{"name": "admin.tools.list"}, {"name": "admin_tools_list"}],
                }
            }
        ),
        encoding="utf-8",
    )
    out = io.StringIO()
    rc = mcp_pin_command(
        str(tmp_path / "m.json"),
        from_file=str(dump),
        claude_plugin=[PLUGIN_SERVER],
        stdout=out,
    )
    assert rc == 2 and "ambiguous callable identity" in out.getvalue()
