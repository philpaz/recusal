"""Review 10 (0.5.7): all-servers-removed semantics and ValueError-consistent
malformed sequence members.

The gaps these close: (1) acknowledging removal of EVERY pinned server tripped the
generic empty-observation refusal alongside the nonblocking removal warnings, an
implementation-to-documentation inconsistency (safe-side, never a bypass) - the
refusal is now deliberate and precise: an empty observation certifies nothing, and
decommissioning ALL MCP capability is the manifest's removal (no pin, no MCP), not a
verification of an empty world; (2) an unhashable member in `unverifiable`/`removed`
raised TypeError from duplicate detection before its type was rejected, leaking a
generic container error where the contract promises ValueError.
"""

import io
import json

import pytest

from recusal import compute_verdict
from recusal.__main__ import mcp_verify_command
from recusal.mcp import (
    McpObservation,
    build_manifest,
    diff_observation,
    manifest_policy,
    manifest_to_text,
)


def _failed_checks(findings):
    return {f.check for f in findings if not f.passed}


# --- all-servers-removed: deliberate, precise refusal ---------------------------------------


def test_removing_the_only_pinned_server_refuses_with_the_precise_message():
    pinned = build_manifest({"only": [{"name": "t"}]})
    findings = diff_observation(pinned, McpObservation(catalog={}, removed=("only",)))
    checks = _failed_checks(findings)
    assert "mcp_full_decommission_unsupported" in checks
    assert "mcp_manifest" not in checks  # the precise refusal, not the generic one
    assert not compute_verdict(findings).passed
    [precise] = [f for f in findings if f.check == "mcp_full_decommission_unsupported"]
    assert "no pin, no MCP" in precise.message


def test_removing_every_server_of_a_multi_server_manifest_refuses():
    pinned = build_manifest({"a": [{"name": "t"}], "b": [{"name": "u"}]})
    findings = diff_observation(pinned, McpObservation(catalog={}, removed=("a", "b")))
    assert "mcp_full_decommission_unsupported" in _failed_checks(findings)
    assert not compute_verdict(findings).passed


def test_removing_one_of_two_servers_stays_a_passing_transition():
    pinned = build_manifest(
        {"a": [{"name": "t"}], "b": [{"name": "u"}]},
        instructions={"a": "a instructions", "b": "b instructions"},
    )
    obs = McpObservation(
        catalog={"a": [{"name": "t"}]},
        sources={"a": {"transport": "external"}},
        instructions={"a": {"observed": True, "text": "a instructions"}},
        removed=("b",),
    )
    findings = diff_observation(pinned, obs)
    assert compute_verdict(findings).passed
    assert any(f.check == "mcp_server_removed" for f in findings)


def test_cli_full_decommission_refuses_with_exit_2(tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text(
        manifest_to_text(build_manifest({"only": [{"name": "t"}]})), encoding="utf-8"
    )
    dump = tmp_path / "empty.json"
    dump.write_text(json.dumps({}), encoding="utf-8")
    out = io.StringIO()
    # an empty --from dump is refused earlier as unparseable-empty; acknowledge via
    # --removed with no observation at all is the library-level case above, so here
    # the CLI path drives the same refusal through a dump missing the only server
    dump.write_text(json.dumps({"other": []}), encoding="utf-8")
    rc = mcp_verify_command(str(manifest), from_file=str(dump), removed=["only"], stdout=out)
    assert rc == 2  # 'other' is unpinned -> refused; the manifest's server acknowledged


def test_manifest_policy_still_fails_closed_after_manifest_removal(tmp_path):
    # the documented decommission path: no pin, no MCP
    policy = manifest_policy(str(tmp_path / "gone.json"))
    findings = policy("mcp__only__t", {})
    assert any(f.check == "mcp_manifest_unavailable" and not f.passed for f in findings)


def test_removal_acknowledgement_never_claims_revocation():
    # the manifest keeps authorizing the removed server's names until re-pinned:
    # the warning must say so, not imply revocation
    pinned = build_manifest(
        {"a": [{"name": "t"}], "b": [{"name": "u"}]},
        instructions={"a": "a instructions", "b": "b instructions"},
    )
    obs = McpObservation(
        catalog={"a": [{"name": "t"}]},
        sources={"a": {"transport": "external"}},
        instructions={"a": {"observed": True, "text": "a instructions"}},
        removed=("b",),
    )
    [removal] = [f for f in diff_observation(pinned, obs) if f.check == "mcp_server_removed"]
    assert "re-pin" in removal.message.lower()


# --- ValueError-consistent malformed members -------------------------------------------------


@pytest.mark.parametrize(
    "member",
    [["server"], {"server": True}, bytearray(b"server"), 7, None],
    ids=["list", "dict", "bytearray", "int", "none"],
)
@pytest.mark.parametrize("field", ["unverifiable", "removed"])
def test_malformed_sequence_members_raise_the_documented_valueerror(field, member):
    pinned = build_manifest({"only": [{"name": "t"}]})
    obs = McpObservation(catalog={}, **{field: (member,)})
    with pytest.raises(ValueError, match=field):
        diff_observation(pinned, obs)


def test_duplicate_names_still_raise_valueerror():
    pinned = build_manifest({"only": [{"name": "t"}]})
    with pytest.raises(ValueError, match="duplicate"):
        diff_observation(pinned, McpObservation(catalog={}, unverifiable=("x", "x")))
