"""Manifest v2 launch identity: a changed server command is refused WITHOUT executing.

The trust gap this closes: v1 pinned the declared catalog only, so `verify` had to run
the configured command to learn the catalog had drifted - by which time a substituted
command had already executed. v2 pins the launch specification (unexpanded template,
args, cwd, env variable NAMES) and compares it BEFORE any process starts.

The flagship proof is the marker test: the attacker swaps the approved command for one
that drops a marker file when executed; verify must exit 2 naming launch-spec drift, and
the marker file must not exist.
"""

import io
import json
import os
import sys

import pytest

from recusal.__main__ import mcp_pin_command, mcp_verify_command
from recusal.mcp import (
    MANIFEST_VERSION,
    build_manifest,
    diff_source,
    load_manifest,
    manifest_to_text,
    normalize_source,
)

SAFE_SERVER = r"""
import json, sys

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method, rid = msg.get("method"), msg.get("id")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": "2025-06-18",
             "capabilities": {"tools": {}}, "serverInfo": {"name": "safe", "version": "0"}}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": rid, "result": {"tools": [
            {"name": "safe_tool", "description": "Reads things.",
             "inputSchema": {"type": "object"}}]}})
"""

# Executed = compromised: this stand-in for a malicious server proves execution by
# writing a marker file. Verification must refuse BEFORE this ever runs.
MARKER_WRITER = r"""
import os, sys
open(os.environ["MARKER_PATH"], "w").write("EXECUTED")
"""


def _write_config(tmp_path, command_args):
    config = tmp_path / ".mcp.json"
    config.write_text(
        json.dumps({"mcpServers": {"srv": {"command": command_args[0], "args": command_args[1:]}}}),
        encoding="utf-8",
    )
    return str(config)


@pytest.fixture()
def arena(tmp_path, monkeypatch):
    safe = tmp_path / "safe_server.py"
    safe.write_text(SAFE_SERVER, encoding="utf-8")
    attacker = tmp_path / "marker_writer.py"
    attacker.write_text(MARKER_WRITER, encoding="utf-8")
    marker = tmp_path / "EXECUTED.marker"
    monkeypatch.setenv("MARKER_PATH", str(marker))
    return {
        "tmp": tmp_path,
        "safe": [sys.executable, str(safe)],
        "attacker": [sys.executable, str(attacker)],
        "marker": marker,
        "manifest": str(tmp_path / "mcp-manifest.json"),
    }


def _pin(arena, config, **kwargs):
    out = io.StringIO()
    rc = mcp_pin_command(
        arena["manifest"],
        claude_config=config,
        approve_server_launch=True,
        stdout=out,
        **kwargs,
    )
    return rc, out.getvalue()


def _verify(arena, config):
    out = io.StringIO()
    rc = mcp_verify_command(arena["manifest"], claude_config=config, stdout=out)
    return rc, out.getvalue()


# --- the flagship adversarial arc -----------------------------------------------------------


def test_a_changed_command_is_refused_without_executing_it(arena):
    config = _write_config(arena["tmp"], arena["safe"])
    rc, _ = _pin(arena, config)
    assert rc == 0
    rc, text = _verify(arena, config)
    assert rc == 0, text  # unchanged spec verifies clean

    # the attack: .mcp.json now points at a command that proves execution by marker
    _write_config(arena["tmp"], arena["attacker"])
    rc, text = _verify(arena, config)
    assert rc == 2
    assert "launch specification changed" in text
    assert not arena["marker"].exists(), "the substituted command EXECUTED during verify"


def test_an_unpinned_new_server_is_refused_before_it_launches(arena):
    config = _write_config(arena["tmp"], arena["safe"])
    rc, _ = _pin(arena, config)
    assert rc == 0

    # attacker ADDS a server rather than changing one
    body = json.loads(open(config, encoding="utf-8").read())
    body["mcpServers"]["extra"] = {
        "command": arena["attacker"][0],
        "args": arena["attacker"][1:],
    }
    open(config, "w", encoding="utf-8").write(json.dumps(body))

    rc, text = _verify(arena, config)
    assert rc == 2
    assert "no pinned launch specification" in text
    assert not arena["marker"].exists(), "an unpinned server EXECUTED during verify"


def test_pin_without_approval_refuses_before_anything_executes(arena):
    config = _write_config(arena["tmp"], arena["attacker"])
    out = io.StringIO()
    rc = mcp_pin_command(arena["manifest"], claude_config=config, stdout=out)
    assert rc == 2
    assert "--approve-server-launch" in out.getvalue()
    assert not arena["marker"].exists(), "pin EXECUTED a command without approval"
    assert not os.path.exists(arena["manifest"])


def test_the_pin_records_the_launch_specification(arena):
    config = _write_config(arena["tmp"], arena["safe"])
    _pin(arena, config)
    manifest = load_manifest(arena["manifest"])
    assert manifest["manifest_version"] == MANIFEST_VERSION == 2
    source = manifest["servers"]["srv"]["source"]
    assert source["transport"] == "stdio"
    assert source["command"] == arena["safe"][0]
    assert source["args"] == arena["safe"][1:]
    assert manifest["servers"]["srv"]["source_fingerprint"].startswith("sha256:")


# --- kernel: normalize/diff/validate ---------------------------------------------------------


def test_diff_source_names_the_changed_fields():
    pinned_source = normalize_source(
        {"transport": "stdio", "command": "npx", "args": ["-y", "srv@1.2.3"], "env_keys": ["T"]}
    )
    manifest = build_manifest({"srv": [{"name": "t"}]}, sources={"srv": pinned_source})
    entry = manifest["servers"]["srv"]
    observed = {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "srv@9.9.9"],
        "env_keys": ["T"],
    }
    findings = diff_source("srv", entry, observed)
    assert len(findings) == 1 and not findings[0].passed
    assert findings[0].context["changed_fields"] == ["args"]
    assert "WITHOUT executing" in findings[0].message


def test_diff_source_refuses_a_hand_edited_pin():
    manifest = build_manifest(
        {"srv": [{"name": "t"}]},
        sources={"srv": {"transport": "stdio", "command": "npx"}},
    )
    entry = dict(manifest["servers"]["srv"])
    entry["source"] = dict(entry["source"], command="evil")  # edited without re-fingerprinting
    findings = diff_source("srv", entry, {"transport": "stdio", "command": "evil"})
    assert not findings[0].passed and "corrupt" in findings[0].message


def test_a_v1_manifest_is_refused_with_a_repin_instruction(tmp_path):
    v1 = {
        "manifest_version": 1,
        "servers": {"srv": {"tools": {"t": {"fingerprint": "sha256:" + "0" * 64}}}},
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(v1), encoding="utf-8")
    with pytest.raises(ValueError, match="predates launch-identity pinning"):
        load_manifest(str(path))


def test_normalize_source_rejects_unknown_and_malformed_fields():
    with pytest.raises(ValueError, match="unknown fields"):
        normalize_source({"transport": "stdio", "command": "x", "secret": "no"})
    with pytest.raises(ValueError, match="transport"):
        normalize_source({})
    with pytest.raises(ValueError, match="command"):
        normalize_source({"transport": "stdio"})
    with pytest.raises(ValueError, match="env_keys"):
        normalize_source({"transport": "stdio", "command": "x", "env_keys": [1]})


def test_external_sources_pin_and_roundtrip(tmp_path):
    manifest = build_manifest({"remote": [{"name": "t", "description": "d"}]})
    assert manifest["servers"]["remote"]["source"] == {"transport": "external"}
    path = tmp_path / "m.json"
    path.write_text(manifest_to_text(manifest), encoding="utf-8")
    assert load_manifest(str(path))["manifest_version"] == 2


# --- more adversarial edges ------------------------------------------------------------------


def test_reordered_args_are_a_launch_spec_change(arena):
    # argv order is meaning; ["-y", "srv"] vs ["srv", "-y"] must not be "the same spec"
    config = _write_config(arena["tmp"], arena["safe"])
    _pin(arena, config)
    swapped = [arena["safe"][0], "-X", arena["safe"][1]]  # inject a flag before the script
    _write_config(arena["tmp"], swapped)
    rc, text = _verify(arena, config)
    assert rc == 2 and "launch specification changed" in text


def test_an_added_env_key_is_a_launch_spec_change(arena):
    config = _write_config(arena["tmp"], arena["safe"])
    _pin(arena, config)
    body = json.loads(open(config, encoding="utf-8").read())
    body["mcpServers"]["srv"]["env"] = {"LD_PRELOAD": "/tmp/evil.so"}
    open(config, "w", encoding="utf-8").write(json.dumps(body))
    rc, text = _verify(arena, config)
    assert rc == 2
    assert "env_keys" in text
    assert not arena["marker"].exists()


def test_a_renamed_server_is_refused_not_silently_relearned(arena):
    # rename = the pinned name goes absent AND an unpinned name appears; the unpinned
    # one must be refused BEFORE it launches.
    config = _write_config(arena["tmp"], arena["safe"])
    _pin(arena, config)
    body = json.loads(open(config, encoding="utf-8").read())
    body["mcpServers"]["renamed"] = body["mcpServers"].pop("srv")
    open(config, "w", encoding="utf-8").write(json.dumps(body))
    rc, text = _verify(arena, config)
    assert rc == 2 and "no pinned launch specification" in text


def test_a_changed_env_value_with_an_unchanged_template_verifies(arena, monkeypatch):
    # The DOCUMENTED residual, pinned as a test so it cannot silently become a claim:
    # identity is template-level; the VALUE a referenced variable expands to is the
    # operator's shell, not the config, and is not pinned.
    monkeypatch.setenv("SAFE_SERVER_PATH", arena["safe"][1])
    config = arena["tmp"] / ".mcp.json"
    config.write_text(
        json.dumps(
            {"mcpServers": {"srv": {"command": arena["safe"][0], "args": ["${SAFE_SERVER_PATH}"]}}}
        ),
        encoding="utf-8",
    )
    rc, _ = _pin(arena, str(config))
    assert rc == 0
    rc, _ = _verify(arena, str(config))
    assert rc == 0  # same template, same verdict - even though the value could differ


def test_pin_and_verify_agree_across_stdio_and_claude_config_forms(arena):
    # The same server pinned via --stdio and verified via --claude-config: templates
    # must compare structurally (command + args), not by which flag supplied them.
    command_text = " ".join(part if " " not in part else f'"{part}"' for part in arena["safe"])
    out = io.StringIO()
    rc = mcp_pin_command(
        arena["manifest"],
        stdio=[["srv", command_text]],
        approve_server_launch=True,
        stdout=out,
    )
    assert rc == 0, out.getvalue()
    config = _write_config(arena["tmp"], arena["safe"])
    rc, text = _verify(arena, config)
    # env_keys/cwd match ([] / None both ways); command+args match structurally
    assert rc == 0, text
