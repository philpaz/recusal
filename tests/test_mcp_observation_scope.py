"""observation_scope (manifest v8): an operator-declared claim about WHAT a pin observed.

The scope label is metadata about observation context (a project config, a machine, an
environment), stored top-level and verbatim. Its contract is deliberately narrow and
honest: recusal verifies the label's STABILITY across re-pins - a changed scope is the
named ``mcp_observation_scope_changed`` WARNING, a review signal, never a refusal
under an already-deliberate ``--update`` - and never its truth. ``null`` is the
explicit no-scope-declared weak claim; the member exists in every v8 manifest, never
by omission, and a blank label refuses because it reads as a declared claim while
claiming nothing. v7 manifests are refused with a migration message naming ``--scope``.
"""

import io
import json

import pytest

from recusal.__main__ import mcp_pin_command, mcp_verify_command
from recusal.mcp import (
    MANIFEST_VERSION,
    build_manifest,
    diff_observation_scope,
    load_manifest,
    manifest_to_text,
)

TOOL = {
    "name": "create_issue",
    "description": "Create an issue.",
    "inputSchema": {"type": "object"},
}


def _write(tmp_path, payload, filename="catalog.json"):
    path = tmp_path / filename
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _roundtrip(tmp_path, manifest):
    path = tmp_path / "m.json"
    path.write_text(manifest_to_text(manifest), encoding="utf-8")
    return load_manifest(str(path))


# --- build ---------------------------------------------------------------------------------


def test_the_default_pin_records_the_explicit_null_scope():
    manifest = build_manifest({"srv": [TOOL]})
    assert manifest["manifest_version"] == MANIFEST_VERSION == 8
    assert "observation_scope" in manifest
    assert manifest["observation_scope"] is None


def test_a_scope_label_is_stored_top_level_and_verbatim():
    manifest = build_manifest({"srv": [TOOL]}, observation_scope="project .mcp.json")
    assert manifest["observation_scope"] == "project .mcp.json"


@pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
def test_a_blank_scope_refuses_the_build(bad):
    with pytest.raises(ValueError, match="nonempty label with visible content"):
        build_manifest({"srv": [TOOL]}, observation_scope=bad)


@pytest.mark.parametrize("bad", [7, ["scope"], {"label": "x"}, True])
def test_a_non_string_scope_refuses_the_build(bad):
    with pytest.raises(ValueError, match="null or a string label"):
        build_manifest({"srv": [TOOL]}, observation_scope=bad)


def test_a_unicode_scope_label_round_trips_through_the_stored_bytes(tmp_path):
    label = "проект über-config \U0001f512"
    manifest = build_manifest({"srv": [TOOL]}, observation_scope=label)
    assert _roundtrip(tmp_path, manifest)["observation_scope"] == label


def test_manifest_bytes_are_deterministic_in_both_scope_states():
    catalog = {"srv": [TOOL]}
    assert manifest_to_text(build_manifest(catalog)) == manifest_to_text(build_manifest(catalog))
    assert manifest_to_text(build_manifest(catalog, observation_scope="ci")) == manifest_to_text(
        build_manifest(catalog, observation_scope="ci")
    )
    # the two states differ ONLY in the scope member's value
    null_text = manifest_to_text(build_manifest(catalog))
    labeled_text = manifest_to_text(build_manifest(catalog, observation_scope="ci"))
    assert '"observation_scope": null' in null_text
    assert '"observation_scope": "ci"' in labeled_text


# --- load / validate -----------------------------------------------------------------------


def test_a_manifest_without_the_scope_member_is_refused_never_by_omission(tmp_path):
    manifest = build_manifest({"srv": [TOOL]})
    del manifest["observation_scope"]
    path = tmp_path / "m.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="never by omission"):
        load_manifest(str(path))


@pytest.mark.parametrize("bad, why", [("", "visible content"), (7, "string label")])
def test_a_corrupt_stored_scope_is_refused(tmp_path, bad, why):
    manifest = build_manifest({"srv": [TOOL]})
    manifest["observation_scope"] = bad
    path = tmp_path / "m.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=why):
        load_manifest(str(path))


def test_a_v7_manifest_is_refused_with_the_migration_message(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"manifest_version": 7, "servers": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="observation-scope metadata.*--scope"):
        load_manifest(str(path))


def test_the_v6_refusal_message_is_unchanged_by_the_v8_bump(tmp_path):
    # each version's migration message stays the historical record of what that
    # version lacked; the v8 bump adds v7's, it does not rewrite v6's
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"manifest_version": 6, "servers": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="resolved-executable identity.*--resolve-executable"):
        load_manifest(str(path))


# --- diff_observation_scope ----------------------------------------------------------------


def test_matching_scopes_pass_with_the_boundary_named():
    for value in (None, "ci machine"):
        findings = diff_observation_scope(value, value)
        assert len(findings) == 1 and findings[0].passed
        assert findings[0].check == "mcp_observation_scope"
        assert findings[0].severity.name == "WARNING"
        assert "truth is not" in findings[0].message


@pytest.mark.parametrize(
    "pinned, current",
    [("laptop", "ci"), (None, "ci"), ("laptop", None)],
)
def test_a_changed_scope_is_the_named_warning_not_a_refusal(pinned, current):
    findings = diff_observation_scope(pinned, current)
    assert len(findings) == 1 and not findings[0].passed
    assert findings[0].check == "mcp_observation_scope_changed"
    assert findings[0].severity.name == "WARNING"  # review signal, never CRITICAL


@pytest.mark.parametrize("bad", ["", "  ", 7])
def test_a_malformed_scope_claim_is_not_evidence(bad):
    with pytest.raises(ValueError, match="observation_scope"):
        diff_observation_scope(bad, "ok")
    with pytest.raises(ValueError, match="observation_scope"):
        diff_observation_scope("ok", bad)


# --- CLI: pin --scope, verify display, --update warning ------------------------------------


def _pin(tmp_path, scope=None, update=False, as_json=False):
    source = _write(tmp_path, {"github": [TOOL]})
    manifest = str(tmp_path / "mcp-manifest.json")
    out = io.StringIO()
    rc = mcp_pin_command(
        manifest,
        from_file=source,
        observation_scope=scope,
        update=update,
        as_json=as_json,
        stdout=out,
    )
    return rc, manifest, out.getvalue()


def test_pin_stores_the_scope_and_verify_prints_it(tmp_path):
    rc, manifest, _ = _pin(tmp_path, scope="project .mcp.json")
    assert rc == 0
    assert load_manifest(manifest)["observation_scope"] == "project .mcp.json"
    out = io.StringIO()
    source = _write(tmp_path, {"github": [TOOL]})
    assert mcp_verify_command(manifest, from_file=source, stdout=out) == 0
    assert "observation scope: 'project .mcp.json'" in out.getvalue()


def test_verify_names_the_explicit_null_scope(tmp_path):
    rc, manifest, _ = _pin(tmp_path)
    assert rc == 0
    out = io.StringIO()
    source = _write(tmp_path, {"github": [TOOL]})
    assert mcp_verify_command(manifest, from_file=source, stdout=out) == 0
    assert "observation scope: none declared" in out.getvalue()


def test_a_blank_scope_flag_refuses_before_anything_runs(tmp_path):
    rc, manifest, text = _pin(tmp_path, scope="   ")
    assert rc == 2
    assert "omit --scope" in text
    assert not (tmp_path / "mcp-manifest.json").exists()


@pytest.mark.parametrize(
    "first, second",
    [("laptop", "ci"), (None, "ci"), ("laptop", None)],
)
def test_repinning_with_a_different_scope_warns_and_pins(tmp_path, first, second):
    rc, manifest, _ = _pin(tmp_path, scope=first)
    assert rc == 0
    rc2, _, text2 = _pin(tmp_path, scope=second, update=True)
    assert rc2 == 0  # WARNING semantics: deliberate --update is never refused for scope
    assert "mcp_observation_scope_changed" in text2
    assert load_manifest(manifest)["observation_scope"] == second


def test_repinning_with_the_same_scope_does_not_warn(tmp_path):
    _pin(tmp_path, scope="ci")
    source = _write(tmp_path, {"github": [TOOL, {**TOOL, "name": "other"}]}, "changed.json")
    manifest = str(tmp_path / "mcp-manifest.json")
    out = io.StringIO()
    rc = mcp_pin_command(
        manifest,
        from_file=source,
        observation_scope="ci",
        update=True,
        stdout=out,
    )
    assert rc == 0
    assert "mcp_observation_scope_changed" not in out.getvalue()


def test_the_json_payload_carries_the_scope_change(tmp_path):
    _pin(tmp_path, scope="laptop")
    rc, _, text = _pin(tmp_path, scope="ci", update=True, as_json=True)
    assert rc == 0
    payload = json.loads(text)
    assert payload["scope_change"] is not None
    assert "laptop" in payload["scope_change"] and "ci" in payload["scope_change"]


def test_replacing_an_identical_scope_manifest_is_still_a_no_op(tmp_path):
    rc, _, _ = _pin(tmp_path, scope="ci")
    assert rc == 0
    rc2, _, text2 = _pin(tmp_path, scope="ci")
    assert rc2 == 0 and "no change" in text2
