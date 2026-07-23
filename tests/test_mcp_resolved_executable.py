"""Manifest v7 strict launch identity: the resolved executable is pinned, not just the template.

The residual this closes (named since 0.5.0): the pin stored the UNEXPANDED command
template, so a swap of the file that template resolves to - a replaced binary on PATH,
edited wrapper-script bytes - passed verify with the template byte-identical. Strict
mode (``pin --resolve-executable``) pins the ``{path, sha256}`` of the first process
image and verify refuses when either changes. A ``null`` pin is the explicit weak
claim: template-only identity, residual as documented. The flagship proof is the
wrapper arc: pin a working wrapper executable strictly, append one line to it (the
template never changes), and verify must exit 2 naming the changed file.
"""

import hashlib
import io
import json
import os
import sys

import pytest

from recusal.__main__ import mcp_pin_command, mcp_verify_command
from recusal.mcp import (
    MANIFEST_VERSION,
    McpObservation,
    build_manifest,
    diff_observation,
    diff_resolved_executable,
    load_manifest,
    manifest_to_text,
)
from recusal.mcp_fetch import McpFetchError, resolve_executable_identity

GOOD = {"path": "/opt/srv", "sha256": "sha256:" + "ab" * 32}
OTHER = {"path": "/opt/srv", "sha256": "sha256:" + "cd" * 32}

TOOL = {"name": "t", "description": "d", "inputSchema": {"type": "object"}}

SERVER_PY = r"""
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
             "capabilities": {"tools": {}}, "serverInfo": {"name": "s", "version": "0"}}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": rid, "result": {"tools": [
            {"name": "safe_tool", "description": "Reads things.",
             "inputSchema": {"type": "object"}}]}})
"""


def _stdio_source(command="py"):
    return {"transport": "stdio", "command": command, "args": [], "cwd": None, "env_templates": {}}


def _strict_manifest(record=GOOD):
    return build_manifest(
        {"srv": [TOOL]}, sources={"srv": _stdio_source()}, resolved_executables={"srv": record}
    )


# ----------------------------------------------------------------- build


def test_a_strict_pin_stores_the_identity_and_others_stay_null():
    manifest = build_manifest(
        {"srv": [TOOL], "other": [TOOL]},
        sources={"srv": _stdio_source(), "other": _stdio_source()},
        resolved_executables={"srv": GOOD},
    )
    assert manifest["servers"]["srv"]["resolved_executable"] == GOOD
    assert manifest["servers"]["other"]["resolved_executable"] is None


def test_an_identity_on_a_non_stdio_source_is_contradictory():
    with pytest.raises(ValueError, match="contradicts a external source"):
        build_manifest({"srv": [TOOL]}, resolved_executables={"srv": GOOD})


def test_an_identity_for_an_unknown_server_refuses():
    with pytest.raises(ValueError, match="not in the catalog"):
        build_manifest(
            {"srv": [TOOL]}, sources={"srv": _stdio_source()}, resolved_executables={"x": GOOD}
        )


@pytest.mark.parametrize(
    "record",
    [
        {"path": "/opt/srv"},
        {"path": "/opt/srv", "sha256": "sha256:" + "ab" * 32, "extra": 1},
        {"path": "", "sha256": "sha256:" + "ab" * 32},
        {"path": "/opt/srv", "sha256": "ab" * 32},
        {"path": "/opt/srv", "sha256": "sha256:" + "AB" * 32},
        "not-a-record",
    ],
)
def test_a_malformed_identity_record_refuses_the_pin(record):
    with pytest.raises(ValueError, match="resolved-executable"):
        build_manifest(
            {"srv": [TOOL]}, sources={"srv": _stdio_source()}, resolved_executables={"srv": record}
        )


# ----------------------------------------------------------------- load


def test_a_v7_entry_must_carry_the_member_explicitly(tmp_path):
    manifest = _strict_manifest()
    del manifest["servers"]["srv"]["resolved_executable"]
    path = tmp_path / "m.json"
    path.write_text(manifest_to_text(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="no resolved_executable member"):
        load_manifest(str(path))


def test_a_hand_edited_identity_refuses_at_load(tmp_path):
    manifest = _strict_manifest()
    manifest["servers"]["srv"]["resolved_executable"] = {"path": "/opt/srv", "sha256": "nope"}
    path = tmp_path / "m.json"
    path.write_text(manifest_to_text(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256:<64 lowercase hex>"):
        load_manifest(str(path))


def test_a_v6_manifest_is_refused_with_the_migration_message(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"manifest_version": 6, "servers": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="resolved-executable identity.*--resolve-executable"):
        load_manifest(str(path))
    assert MANIFEST_VERSION == 8


# ----------------------------------------------------------------- diff


def _entry(record):
    return _strict_manifest(record)["servers"]["srv"] if record is not None else None


def test_a_null_pin_adjudicates_nothing():
    manifest = build_manifest({"srv": [TOOL]}, sources={"srv": _stdio_source()})
    assert diff_resolved_executable("srv", manifest["servers"]["srv"], GOOD) == []


def test_a_matching_identity_is_affirmative():
    findings = diff_resolved_executable("srv", _entry(GOOD), dict(GOOD))
    assert [f.check for f in findings] == ["mcp_resolved_executable"]
    assert findings[0].passed


def test_changed_bytes_refuse_naming_the_field():
    findings = diff_resolved_executable("srv", _entry(GOOD), OTHER)
    assert [f.check for f in findings] == ["mcp_resolved_executable_changed"]
    assert not findings[0].passed and findings[0].severity.value == "CRITICAL"
    assert findings[0].context["changed_fields"] == ["sha256"]


def test_a_changed_path_refuses_too():
    moved = {"path": "/usr/bin/srv", "sha256": GOOD["sha256"]}
    findings = diff_resolved_executable("srv", _entry(GOOD), moved)
    assert findings[0].context["changed_fields"] == ["path"]


def test_an_explicit_resolution_failure_refuses():
    findings = diff_resolved_executable("srv", _entry(GOOD), None)
    assert [f.check for f in findings] == ["mcp_resolved_executable_unverifiable"]
    assert not findings[0].passed and findings[0].severity.value == "CRITICAL"


def test_a_corrupt_pin_record_refuses():
    entry = _entry(GOOD)
    entry["resolved_executable"] = {"path": "/opt/srv"}
    findings = diff_resolved_executable("srv", entry, GOOD)
    assert [f.check for f in findings] == ["mcp_resolved_executable_corrupt"]


def test_a_malformed_observation_is_not_evidence():
    with pytest.raises(ValueError, match="resolved-executable observation"):
        diff_resolved_executable("srv", _entry(GOOD), {"path": "/opt/srv"})


# ------------------------------------------------- diff_observation composition


def _observation(**overrides):
    base = dict(
        catalog={"srv": [TOOL]},
        sources={"srv": _stdio_source()},
        instructions={"srv": {"observed": False, "text": None}},
    )
    base.update(overrides)
    return McpObservation(**base)


def test_a_strict_pin_with_no_resolved_component_is_a_critical_omission():
    findings = diff_observation(_strict_manifest(), _observation())
    checks = [f.check for f in findings if not f.passed]
    assert "mcp_resolved_executable_unobserved" in checks


def test_a_strict_pin_verifies_clean_with_a_matching_component():
    findings = diff_observation(
        _strict_manifest(), _observation(resolved_executables={"srv": dict(GOOD)})
    )
    assert all(f.passed for f in findings)
    assert "mcp_resolved_executable" in [f.check for f in findings]


def test_a_null_pin_needs_no_resolved_component():
    manifest = build_manifest({"srv": [TOOL]}, sources={"srv": _stdio_source()})
    findings = diff_observation(manifest, _observation())
    assert all(f.passed for f in findings)
    assert not any(f.check.startswith("mcp_resolved_executable") for f in findings)


def test_a_removed_strict_server_needs_no_resolved_component():
    findings = diff_observation(
        _strict_manifest(),
        McpObservation(
            catalog={"other": [TOOL]},
            sources={"other": _stdio_source()},
            instructions={"other": {"observed": False, "text": None}},
            removed=("srv",),
        ),
    )
    # the unpinned 'other' server refuses on its own; the point here is that no
    # resolved-executable finding fires for the deliberately removed strict server
    assert not any(f.check.startswith("mcp_resolved_executable") for f in findings)


def test_an_identity_for_a_server_outside_the_catalog_is_malformed():
    with pytest.raises(ValueError, match="not in the catalog"):
        diff_observation(
            _strict_manifest(), _observation(resolved_executables={"ghost": dict(GOOD)})
        )


# ----------------------------------------------------------------- fetch


def test_resolution_hashes_the_real_resolved_file():
    identity = resolve_executable_identity([sys.executable])
    assert identity["path"] == os.path.abspath(sys.executable)
    with open(sys.executable, "rb") as fh:
        expected = hashlib.sha256(fh.read()).hexdigest()
    assert identity["sha256"] == f"sha256:{expected}"


def test_an_unresolvable_command_raises():
    with pytest.raises(McpFetchError, match="not found on PATH"):
        resolve_executable_identity(["definitely-not-a-real-command-xyz"])


# ------------------------------------------------------- CLI live arc (flagship)


def _make_wrapper(tmp_path, server_py):
    """A standalone executable whose bytes can drift while the template cannot."""
    if os.name == "nt":
        wrapper = tmp_path / "wrapper.bat"
        wrapper.write_text(f'@echo off\r\n"{sys.executable}" "{server_py}" %*\r\n')
    else:
        wrapper = tmp_path / "wrapper.sh"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{server_py}" "$@"\n')
        wrapper.chmod(0o755)
    return str(wrapper)


@pytest.fixture()
def arena(tmp_path):
    server_py = tmp_path / "server.py"
    server_py.write_text(SERVER_PY, encoding="utf-8")
    return {
        "wrapper": _make_wrapper(tmp_path, server_py),
        "manifest": str(tmp_path / "mcp-manifest.json"),
    }


def _pin(arena, *, resolve=True):
    out = io.StringIO()
    code = mcp_pin_command(
        arena["manifest"],
        stdio=[["srv", f'"{arena["wrapper"]}"']],
        approve_server_launch=True,
        resolve_executable=resolve,
        stdout=out,
    )
    return code, out.getvalue()


def _verify(arena):
    out = io.StringIO()
    code = mcp_verify_command(
        arena["manifest"], stdio=[["srv", f'"{arena["wrapper"]}"']], stdout=out
    )
    return code, out.getvalue()


def test_the_wrapper_arc_pin_verify_drift_refuse(arena):
    code, text = _pin(arena)
    assert code == 0, text
    pinned = load_manifest(arena["manifest"])["servers"]["srv"]["resolved_executable"]
    assert pinned is not None
    assert pinned["path"] == os.path.abspath(arena["wrapper"])

    code, text = _verify(arena)
    assert code == 0, text
    assert "[ok] mcp_resolved_executable:" in text

    # The drift: one appended line. The command template is byte-identical; only
    # the file it resolves to changed. Template-only pins pass this; strict refuses.
    with open(arena["wrapper"], "a", encoding="utf-8") as fh:
        fh.write("rem drift\r\n" if os.name == "nt" else "# drift\n")
    code, text = _verify(arena)
    assert code == 2, text
    assert "mcp_resolved_executable_changed" in text


def test_a_template_only_pin_still_passes_the_same_drift(arena):
    code, text = _pin(arena, resolve=False)
    assert code == 0, text
    assert load_manifest(arena["manifest"])["servers"]["srv"]["resolved_executable"] is None
    with open(arena["wrapper"], "a", encoding="utf-8") as fh:
        fh.write("rem drift\r\n" if os.name == "nt" else "# drift\n")
    # the honest boundary, pinned as a test: null = template-only, drift passes
    code, text = _verify(arena)
    assert code == 0, text
    assert "mcp_resolved_executable" not in text


def test_resolve_executable_needs_a_launching_source(tmp_path):
    dump = tmp_path / "dump.json"
    dump.write_text(json.dumps({"srv": [TOOL]}), encoding="utf-8")
    out = io.StringIO()
    code = mcp_pin_command(
        str(tmp_path / "m.json"),
        from_file=str(dump),
        resolve_executable=True,
        stdout=out,
    )
    assert code == 2
    assert "no executable to resolve" in out.getvalue()
