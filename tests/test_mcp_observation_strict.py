"""Review 8 (0.5.5): the complete-observation contract, malformed-event provenance
reset, canonical tool-pin fields, and the WebSocket header-only boundary.

The trust gaps these close: (1) `diff_observation` compared only the components the
caller supplied, so omitting `sources` bypassed launch/remote identity entirely and
component-only servers escaped adjudication; (2) a malformed hook event never reached
the policy, so in a reused process its audit record inherited the previous valid
adjudication's manifest digest; (3) a tool pin could omit `fields` or carry undefined
diagnostic names; (4) the parser accepted OAuth for WebSocket entries, a shape Claude
documents as unsupported (WebSocket authentication is header-only).
"""

import io
import json

import pytest

from recusal import compute_verdict
from recusal.mcp import (
    McpObservation,
    build_manifest,
    diff_observation,
    load_manifest,
    manifest_policy,
    manifest_to_text,
    normalize_source,
)
from recusal.mcp_fetch import servers_from_claude_config

CATALOG = {"srv": [{"name": "t", "description": "safe"}]}


def _pinned(instructions=None):
    return build_manifest(CATALOG, instructions=instructions)


def _failed_checks(findings):
    return {f.check for f in findings if not f.passed}


def _write_manifest(tmp_path):
    import hashlib

    path = tmp_path / "m.json"
    path.write_text(manifest_to_text(build_manifest(CATALOG)), encoding="utf-8")
    return str(path), "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _stdio_pinned():
    return build_manifest(
        CATALOG,
        sources={"srv": {"transport": "stdio", "command": "py", "args": ["server.py"]}},
        instructions={"srv": "approved instructions"},
    )


def _http_pinned():
    return build_manifest(
        CATALOG,
        sources={
            "srv": {
                "transport": "http",
                "url_template": "${MCP_URL}",
                "header_templates": {"Authorization": "Bearer ${TOKEN}"},
            }
        },
        instructions={"srv": "approved instructions"},
    )


def _full_instructions(text="approved instructions"):
    return {"srv": {"observed": True, "text": text}}


# --- the complete-observation contract -------------------------------------------------------


@pytest.mark.parametrize("pinned_factory", [_stdio_pinned, _http_pinned])
def test_a_pinned_source_cannot_pass_when_sources_is_omitted(pinned_factory):
    # the source-omission bypass (review 8 P0-1): matching catalog + matching
    # instructions, sources map left out entirely - must refuse, never silently
    # verify catalog-only against a pin that carries launch/remote identity
    obs = McpObservation(catalog=CATALOG, instructions=_full_instructions())
    findings = diff_observation(pinned_factory(), obs)
    assert "mcp_source_unobserved" in _failed_checks(findings)
    assert not compute_verdict(findings).passed


def test_a_changed_source_cannot_be_bypassed_by_omitting_the_source_map():
    # with sources SUPPLIED and drifted the verify names the drift; dropping the map
    # must not do better than supplying the truth
    pinned = _stdio_pinned()
    drifted = McpObservation(
        catalog=CATALOG,
        sources={"srv": {"transport": "stdio", "command": "py", "args": ["EVIL.py"]}},
        instructions=_full_instructions(),
    )
    assert "mcp_launch_spec_changed" in _failed_checks(diff_observation(pinned, drifted))
    omitted = McpObservation(catalog=CATALOG, instructions=_full_instructions())
    assert not compute_verdict(diff_observation(pinned, omitted)).passed


def test_unpinned_component_only_servers_are_refused():
    pinned = _pinned(instructions={"srv": "approved instructions"})
    base = dict(
        catalog=CATALOG,
        sources={"srv": {"transport": "external"}},
        instructions=_full_instructions(),
    )
    with_source = McpObservation(
        catalog=base["catalog"],
        sources={**base["sources"], "ghost": {"transport": "external"}},
        instructions=base["instructions"],
    )
    with_instructions = McpObservation(
        catalog=base["catalog"],
        sources=base["sources"],
        instructions={**base["instructions"], "ghost": {"observed": False, "text": None}},
    )
    with_unverifiable = McpObservation(
        catalog=base["catalog"],
        sources=base["sources"],
        instructions=base["instructions"],
        unverifiable=("ghost",),
    )
    for obs in (with_source, with_instructions, with_unverifiable):
        findings = diff_observation(pinned, obs)
        ghosts = [f for f in findings if not f.passed and f.context.get("server") == "ghost"]
        assert ghosts and ghosts[0].check == "mcp_unpinned_server"


def test_a_pinned_source_only_server_without_catalog_is_incomplete_not_absent():
    pinned = _stdio_pinned()
    obs = McpObservation(
        catalog={},
        sources={"srv": {"transport": "stdio", "command": "py", "args": ["server.py"]}},
        instructions=_full_instructions(),
    )
    findings = diff_observation(pinned, obs)
    assert "mcp_observation_incomplete" in _failed_checks(findings)
    # ...and naming it unverifiable is the legitimate representation instead
    named = McpObservation(catalog={}, unverifiable=("srv",))
    checks = _failed_checks(diff_observation(pinned, named))
    assert "mcp_pinned_server_unverifiable" in checks
    assert "mcp_observation_incomplete" not in checks


@pytest.mark.parametrize(
    "record",
    [
        {"observed": "false", "text": None},  # truthy string is NOT observation
        {"observed": 1, "text": None},  # int is not bool
        {"observed": True, "text": None, "note": "x"},
        {"observed": True},  # missing text member
        {"observed": False, "text": "left over"},
        {"observed": True, "text": 42},
    ],
)
def test_malformed_instruction_observations_are_rejected_not_coerced(record):
    pinned = _pinned(instructions={"srv": "approved instructions"})
    obs = McpObservation(
        catalog=CATALOG,
        sources={"srv": {"transport": "external"}},
        instructions={"srv": record},
    )
    with pytest.raises(ValueError):
        diff_observation(pinned, obs)


@pytest.mark.parametrize("bad_tools", [None, "", {}, 7])
def test_malformed_catalog_values_refuse_in_the_unified_api(bad_tools):
    pinned = _pinned(instructions={"srv": "approved instructions"})
    obs = McpObservation(catalog={"srv": bad_tools})
    with pytest.raises(ValueError, match="must refuse, never normalize"):
        diff_observation(pinned, obs)


def test_duplicate_or_invalid_unverifiable_names_are_rejected():
    pinned = _pinned()
    for bad in (("srv", "srv"), ("",), (None,)):
        with pytest.raises(ValueError, match="unverifiable"):
            diff_observation(pinned, McpObservation(catalog={}, unverifiable=bad))


# --- malformed events never inherit manifest provenance -------------------------------------


def _hook_once(policy, payload, audit=None, fail_closed=True):
    from recusal.claude_code import run_pretooluse_hook

    return run_pretooluse_hook(
        policy,
        audit=audit,
        fail_closed=fail_closed,
        stdin=io.StringIO(payload),
        stdout=io.StringIO(),
    )


MALFORMED_EVENTS = [
    "{not json",  # unparseable
    '"just a string"',  # non-object JSON
    '{"tool_input": {}}',  # missing tool_name
    '{"tool_name": "", "tool_input": {}}',  # empty tool_name
    '{"tool_name": 3, "tool_input": {}}',  # non-string tool_name
    '{"tool_name": "mcp__srv__t", "tool_input": []}',  # non-object tool_input
]


@pytest.mark.parametrize("malformed", MALFORMED_EVENTS)
@pytest.mark.parametrize("fail_closed", [True, False])
def test_a_malformed_event_never_inherits_the_previous_digest(tmp_path, malformed, fail_closed):
    # one process, one thread, one policy object (review 8 P0-2): a valid MCP event
    # leaves the verified digest in this context's ContextVar; the malformed event
    # that follows never reaches the policy, so the hook itself must reset before
    # parsing or the malformed record inherits provenance it never had
    from recusal.audit import AuditLog

    manifest_path, digest = _write_manifest(tmp_path)
    policy = manifest_policy(manifest_path)
    log = tmp_path / "audit.jsonl"

    _hook_once(policy, '{"tool_name": "mcp__srv__t", "tool_input": {}}', audit=AuditLog(str(log)))
    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert entries[-1]["action"]["control"]["manifest_sha256"] == digest  # the valid one

    _hook_once(policy, malformed, audit=AuditLog(str(log)), fail_closed=fail_closed)
    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    malformed_control = entries[-1]["action"]["control"]
    assert "manifest_sha256" not in malformed_control, (
        "an event that never reached the policy inherited the previous manifest digest"
    )


def test_valid_unaudited_call_then_malformed_audited_call_has_no_stale_digest(tmp_path):
    from recusal.audit import AuditLog

    manifest_path, _digest = _write_manifest(tmp_path)
    policy = manifest_policy(manifest_path)
    _hook_once(policy, '{"tool_name": "mcp__srv__t", "tool_input": {}}')  # no audit
    log = tmp_path / "audit.jsonl"
    _hook_once(policy, "{not json", audit=AuditLog(str(log)))
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert "manifest_sha256" not in entry["action"]["control"]


# --- canonical tool-pin fields ---------------------------------------------------------------


def test_a_tool_pin_without_fields_is_not_canonical(tmp_path):
    manifest = build_manifest(CATALOG)
    del manifest["servers"]["srv"]["tools"]["t"]["fields"]
    path = tmp_path / "m.json"
    path.write_text(manifest_to_text(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical"):
        load_manifest(str(path))


def test_a_tool_pin_with_an_undefined_diagnostic_field_is_refused(tmp_path):
    manifest = build_manifest(CATALOG)
    manifest["servers"]["srv"]["tools"]["t"]["fields"]["undefinedField"] = "sha256:" + "0" * 64
    path = tmp_path / "m.json"
    path.write_text(manifest_to_text(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="tracked diagnostic set"):
        load_manifest(str(path))


# --- WebSocket is header-only, per the documented Claude surface ----------------------------


def test_ws_oauth_refuses_in_config_parsing(tmp_path):
    config = tmp_path / ".mcp.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "wsrv": {
                        "type": "ws",
                        "url": "wss://example.test/mcp",
                        "oauth": {"scopes": "read"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="header-only"):
        servers_from_claude_config(str(config))


def test_ws_headers_and_helper_still_parse_and_carry_no_oauth_member(tmp_path):
    config = tmp_path / ".mcp.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "wsrv": {
                        "type": "ws",
                        "url": "wss://example.test/mcp",
                        "headers": {"Authorization": "Bearer ${WS_TOKEN}"},
                        "headersHelper": "get-ws-token.sh",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    _, remote = servers_from_claude_config(str(config))
    source = remote["wsrv"]
    assert "oauth" not in source  # canonical ws shape has NO oauth member at all
    assert "oauth" not in normalize_source(source)


def test_http_oauth_remains_supported(tmp_path):
    config = tmp_path / ".mcp.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "hsrv": {
                        "type": "http",
                        "url": "https://example.test/mcp",
                        "oauth": {"clientId": "abc", "scopes": "read write"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    _, remote = servers_from_claude_config(str(config))
    assert remote["hsrv"]["oauth"]["scopes"] == "read write"


def test_normalize_source_refuses_ws_oauth_directly():
    with pytest.raises(ValueError, match="header-only"):
        normalize_source(
            {"transport": "ws", "url_template": "wss://example.test", "oauth": {"scopes": "read"}}
        )
