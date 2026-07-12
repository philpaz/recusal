"""diff_observation is the one omission-resistant manifest-v5 verify (review 7, P0-1),
rich single-server --from observations keep their instructions (P0-2), and manifest
audit provenance is invocation-local under CONCURRENT policy reuse (P0-3).

The trust gaps these close: (1) `diff_manifest` alone is catalog-only, so a
programmatic caller could verify unchanged tools and read a clean result as a full v5
verify while the server's instructions had been rewritten; (2) the `--server` branch
recorded a supplied `instructions` field as observed:false, silently downgrading a
stronger observation to the weaker claim; (3) the manifest digest rode on a shared
mutable attribute, so two threads sharing one policy object could cross-contaminate
each other's audit provenance.
"""

import io
import json
import threading

import pytest

from recusal import compute_verdict
from recusal.__main__ import mcp_pin_command, mcp_verify_command
from recusal.claude_code import _control_identity
from recusal.mcp import (
    McpObservation,
    build_manifest,
    diff_manifest,
    diff_observation,
    load_manifest,
    manifest_policy,
    manifest_to_text,
)

CATALOG = {"srv": [{"name": "t", "description": "safe"}]}


def _pinned(instructions=None):
    return build_manifest(CATALOG, instructions=instructions)


def _failed_checks(findings):
    return {f.check for f in findings if not f.passed}


# --- P0-1: the unified API cannot be satisfied by a catalog-only observation ----------------


def test_changed_instructions_cannot_pass_the_unified_api():
    pinned = _pinned(instructions={"srv": "approved instructions"})
    obs = McpObservation(
        catalog=CATALOG,
        instructions={"srv": {"observed": True, "text": "REWRITTEN instructions"}},
    )
    findings = diff_observation(pinned, obs)
    assert "mcp_instructions_changed" in _failed_checks(findings)
    assert not compute_verdict(findings).passed


def test_missing_instruction_observation_refuses_when_the_pin_covered_them():
    # the omission-resistance property itself: leaving instructions out of the
    # observation must be a refusal, never a silently weaker verify
    pinned = _pinned(instructions={"srv": "approved instructions"})
    findings = diff_observation(pinned, McpObservation(catalog=CATALOG))
    assert "mcp_instructions_unobserved" in _failed_checks(findings)
    assert not compute_verdict(findings).passed


def test_matching_instructions_pass_the_unified_api():
    pinned = _pinned(instructions={"srv": "approved instructions"})
    obs = McpObservation(
        catalog=CATALOG,
        instructions={"srv": {"observed": True, "text": "approved instructions"}},
    )
    verdict = compute_verdict(diff_observation(pinned, obs))
    assert verdict.passed


def test_legacy_unobserved_pin_stays_explicitly_weaker_not_blocking():
    pinned = _pinned()  # no instructions observed at pin
    findings = diff_observation(pinned, McpObservation(catalog=CATALOG))
    assert compute_verdict(findings).passed
    boundary = [f for f in findings if f.check == "mcp_instructions" and f.passed]
    assert boundary and "does not cover" in boundary[0].message


def test_diff_manifest_alone_is_catalog_only_and_documented_as_such():
    # the boundary the unified API exists to close, pinned as behavior: catalog-only
    # comparison CANNOT see an instruction rewrite (that is exactly why callers must
    # use diff_observation for a complete v5 verify)
    pinned = _pinned(instructions={"srv": "approved instructions"})
    assert compute_verdict(diff_manifest(pinned, CATALOG)).passed
    assert "catalog-only" in (diff_manifest.__doc__ or "").lower()


def test_cli_and_unified_api_agree_on_blocking(tmp_path):
    manifest = tmp_path / "m.json"
    dump = tmp_path / "obs.json"
    rich = {"srv": {"instructions": "approved instructions", "tools": CATALOG["srv"]}}
    dump.write_text(json.dumps(rich), encoding="utf-8")
    assert mcp_pin_command(str(manifest), from_file=str(dump), stdout=io.StringIO()) == 0

    drifted = {"srv": {"instructions": "REWRITTEN", "tools": CATALOG["srv"]}}
    dump.write_text(json.dumps(drifted), encoding="utf-8")
    out = io.StringIO()
    rc = mcp_verify_command(str(manifest), from_file=str(dump), stdout=out)
    assert rc == 2 and "mcp_instructions_changed" in out.getvalue()

    pinned = load_manifest(str(manifest))
    obs = McpObservation(
        catalog=CATALOG,
        sources={"srv": {"transport": "external"}},
        instructions={"srv": {"observed": True, "text": "REWRITTEN"}},
    )
    assert "mcp_instructions_changed" in _failed_checks(diff_observation(pinned, obs))


# --- P0-2: rich single-server --from observations keep their instructions -------------------


def _server_dump(tmp_path, body):
    path = tmp_path / "one.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return str(path)


def test_rich_single_server_pin_records_instructions_as_observed(tmp_path):
    manifest = tmp_path / "m.json"
    dump = _server_dump(
        tmp_path, {"instructions": "use only for servicing", "tools": CATALOG["srv"]}
    )
    rc = mcp_pin_command(str(manifest), from_file=dump, server="srv", stdout=io.StringIO())
    assert rc == 0
    record = load_manifest(str(manifest))["servers"]["srv"]["server_instructions"]
    assert record["observed"] is True and record["present"] is True
    assert "servicing" not in manifest.read_text(encoding="utf-8")  # hash, never text


def test_rich_single_server_verify_detects_changed_instructions(tmp_path):
    manifest = tmp_path / "m.json"
    dump = _server_dump(tmp_path, {"instructions": "approved", "tools": CATALOG["srv"]})
    assert mcp_pin_command(str(manifest), from_file=dump, server="srv", stdout=io.StringIO()) == 0
    dump = _server_dump(tmp_path, {"instructions": "REWRITTEN", "tools": CATALOG["srv"]})
    out = io.StringIO()
    rc = mcp_verify_command(str(manifest), from_file=dump, server="srv", stdout=out)
    assert rc == 2 and "changed its instructions" in out.getvalue()


def test_rich_single_server_verify_detects_added_and_removed_instructions(tmp_path):
    manifest = tmp_path / "m.json"
    # pinned observed-with-none (explicit null), then the server ADDS instructions
    dump = _server_dump(tmp_path, {"instructions": None, "tools": CATALOG["srv"]})
    assert mcp_pin_command(str(manifest), from_file=dump, server="srv", stdout=io.StringIO()) == 0
    dump = _server_dump(tmp_path, {"instructions": "sneaked in", "tools": CATALOG["srv"]})
    out = io.StringIO()
    assert mcp_verify_command(str(manifest), from_file=dump, server="srv", stdout=out) == 2
    assert "added instructions" in out.getvalue()

    # and the reverse: pinned WITH instructions, then removed
    manifest2 = tmp_path / "m2.json"
    dump = _server_dump(tmp_path, {"instructions": "approved", "tools": CATALOG["srv"]})
    assert mcp_pin_command(str(manifest2), from_file=dump, server="srv", stdout=io.StringIO()) == 0
    dump = _server_dump(tmp_path, {"instructions": None, "tools": CATALOG["srv"]})
    out = io.StringIO()
    assert mcp_verify_command(str(manifest2), from_file=dump, server="srv", stdout=out) == 2
    assert "removed its pinned instructions" in out.getvalue()


def test_non_string_single_server_instructions_refuse(tmp_path):
    manifest = tmp_path / "m.json"
    dump = _server_dump(tmp_path, {"instructions": 42, "tools": CATALOG["srv"]})
    out = io.StringIO()
    rc = mcp_pin_command(str(manifest), from_file=dump, server="srv", stdout=out)
    assert rc == 2 and "must be a string or null" in out.getvalue()


def test_raw_tools_list_with_server_keeps_the_weaker_claim(tmp_path):
    manifest = tmp_path / "m.json"
    dump = _server_dump(tmp_path, {"tools": CATALOG["srv"]})  # no instructions key
    assert mcp_pin_command(str(manifest), from_file=dump, server="srv", stdout=io.StringIO()) == 0
    record = load_manifest(str(manifest))["servers"]["srv"]["server_instructions"]
    assert record == {"observed": False}


def test_from_help_documents_the_rich_observation_shape():
    import argparse

    from recusal.__main__ import _add_mcp_source_args

    parser = argparse.ArgumentParser()
    _add_mcp_source_args(parser)
    text = parser.format_help()
    # the CLI surface itself must not teach only the weaker legacy shapes
    assert "'instructions': str|null" in text
    assert "observed: false" in text


# --- P1-1: only canonical server_instructions shapes load ----------------------------------


def _manifest_with_record(record):
    manifest = build_manifest(CATALOG)
    manifest["servers"]["srv"]["server_instructions"] = record
    return manifest


@pytest.mark.parametrize(
    "record",
    [
        {"observed": False, "present": True, "fingerprint": "sha256:" + "0" * 64},
        {"observed": False, "present": False},
        {"observed": True, "present": False, "fingerprint": "sha256:" + "0" * 64},
        {"observed": True, "present": True},
        {"observed": True, "present": True, "fingerprint": "sha256:" + "0" * 64, "note": "x"},
        {"observed": False, "extra": 1},
    ],
)
def test_noncanonical_instruction_records_are_refused(tmp_path, record):
    path = tmp_path / "m.json"
    path.write_text(manifest_to_text(_manifest_with_record(record)), encoding="utf-8")
    with pytest.raises(ValueError, match="server_instructions"):
        load_manifest(str(path))


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda m: m.update({"generated_by": "tool"}), "fields this loader does not define"),
        (lambda m: m["servers"]["srv"].update({"note": "x"}), "undefined fields"),
        (
            lambda m: m["servers"]["srv"]["tools"]["t"].update({"reviewed": True}),
            "undefined fields",
        ),
    ],
)
def test_undefined_manifest_fields_are_refused_at_every_level(tmp_path, mutate, match):
    manifest = build_manifest(CATALOG)
    mutate(manifest)
    path = tmp_path / "m.json"
    path.write_text(manifest_to_text(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_manifest(str(path))


def test_builder_output_is_canonical_and_loads():
    # the strictness must never refuse what the builder itself emits
    text = manifest_to_text(
        build_manifest(
            {"a": [{"name": "t"}], "b": [{"name": "u"}], "c": [{"name": "v"}]},
            instructions={"a": "covered", "b": None},
        )
    )
    import json as _json

    from recusal.mcp import _validate_manifest

    _validate_manifest(_json.loads(text))


# --- P0-3: audit manifest provenance is invocation-local under concurrency ------------------


def _write_manifest(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(manifest_to_text(build_manifest(CATALOG)), encoding="utf-8")
    import hashlib

    return str(path), "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_concurrent_invocations_each_see_their_own_digest(tmp_path):
    manifest_path, digest = _write_manifest(tmp_path)
    policy = manifest_policy(manifest_path)

    a_called = threading.Event()
    b_done = threading.Event()
    results = {}

    def thread_a():
        policy("mcp__srv__t", {})
        a_called.set()
        assert b_done.wait(10)
        # B has since run a non-MCP call (which clears ITS context) - A's audit record,
        # built after that interleaving, must still carry A's own digest
        results["a"] = _control_identity(policy, None).get("manifest_sha256")

    def thread_b():
        assert a_called.wait(10)
        policy("not_mcp", {})
        results["b"] = _control_identity(policy, None).get("manifest_sha256")
        b_done.set()

    threads = [threading.Thread(target=thread_a), threading.Thread(target=thread_b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert results["a"] == digest, "invocation A lost its own provenance to B's clear"
    assert results["b"] is None, "a non-MCP invocation inherited another invocation's digest"


def test_concurrent_corrupt_manifest_never_records_a_digest(tmp_path):
    manifest_path, digest = _write_manifest(tmp_path)
    policy = manifest_policy(manifest_path)

    a_called = threading.Event()
    b_done = threading.Event()
    results = {}

    def thread_a():
        policy("mcp__srv__t", {})  # validates the good manifest
        a_called.set()
        assert b_done.wait(10)
        results["a"] = _control_identity(policy, None).get("manifest_sha256")

    def thread_b():
        assert a_called.wait(10)
        with open(manifest_path, "w", encoding="utf-8") as fh:
            fh.write("{corrupt")
        findings = policy("mcp__srv__t", {})
        results["b_refused"] = any(
            f.check == "mcp_manifest_unavailable" for f in findings if not f.passed
        )
        results["b"] = _control_identity(policy, None).get("manifest_sha256")
        b_done.set()

    threads = [threading.Thread(target=thread_a), threading.Thread(target=thread_b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert results["b_refused"], "a corrupt manifest did not refuse"
    assert results["b"] is None, "a corrupt manifest was recorded as enforced"
    assert results["a"] == digest, "the corrupting invocation erased another's provenance"


def test_policy_exposes_get_control_identity_and_legacy_attribute_still_reads():
    # the audit layer prefers the invocation-local getter; a custom policy object using
    # the documented plain attribute (sequential-only seam) must keep working
    class _Custom:
        last_manifest_digest = "sha256:" + "a" * 64

        def __call__(self, tool_name, tool_input):
            return []

    identity = _control_identity(_Custom(), {"policy_id": "x"})
    assert identity["manifest_sha256"] == "sha256:" + "a" * 64
    assert identity["policy_id"] == "x"
