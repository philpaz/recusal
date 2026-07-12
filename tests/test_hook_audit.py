"""``run_pretooluse_hook(audit=...)``: every adjudication on the record, one wire.

The contract: defer, allow, and deny each append one hash-chained entry naming the tool,
the decision, and the reasons; the proposed tool_input is bound by SHA-256 fingerprint,
never embedded (a Write's file body or an env value must not leak into the log); an
unwritable log fails CLOSED to a deny unless fail_closed=False; and a malformed event is
itself on the record.
"""

import io
import json

import pytest

from recusal import AuditLog, Finding
from recusal.audit import load, verify_file
from recusal.claude_code import run_pretooluse_hook


def _deny_rm(tool_name, tool_input):
    if tool_name == "Bash" and "rm -rf" in tool_input.get("command", ""):
        return [Finding.fail("destructive_bash", severity="CRITICAL", message="refusing rm -rf")]
    return []


def _run(event, audit, policy=_deny_rm, **kwargs):
    out = io.StringIO()
    stdin = io.StringIO(event if isinstance(event, str) else json.dumps(event))
    res = run_pretooluse_hook(policy, audit=audit, stdin=stdin, stdout=out, **kwargs)
    return res, out.getvalue()


def test_a_deny_is_on_the_record(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    res, _ = _run({"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}}, AuditLog(path=path))
    assert res["hookSpecificOutput"]["permissionDecision"] == "deny"
    entries = load(path)
    assert len(entries) == 1
    assert entries[0]["decision"] == "FAIL"
    assert entries[0]["action"]["tool"] == "Bash"
    assert entries[0]["action"]["decision"] == "deny"
    assert "destructive_bash" in json.dumps(entries[0]["failures"])
    ok, problems = verify_file(path)
    assert ok, problems


def test_a_defer_is_on_the_record_too(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    res, text = _run({"tool_name": "Read", "tool_input": {"file_path": "x"}}, AuditLog(path=path))
    assert res is None and text == ""  # defer still emits nothing to Claude Code
    entries = load(path)
    assert len(entries) == 1
    assert entries[0]["action"]["decision"] == "defer"
    assert entries[0]["decision"] == "PASS"


def test_tool_input_is_fingerprinted_never_embedded(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    secret = "AKIA-SUPER-SECRET-VALUE"
    _run(
        {"tool_name": "Write", "tool_input": {"file_path": "a.txt", "content": secret}},
        AuditLog(path=path),
    )
    entries = load(path)
    assert len(entries[0]["action"]["input_sha256"]) == 64
    with open(path, encoding="utf-8") as fh:
        assert secret not in fh.read()


def test_the_actor_defaults_to_the_events_session_id(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    _run(
        {"tool_name": "Read", "tool_input": {}, "session_id": "sess-42"},
        AuditLog(path=path),
    )
    assert load(path)[0]["actor"] == "sess-42"


def test_an_explicit_actor_wins_over_the_session_id(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    _run(
        {"tool_name": "Read", "tool_input": {}, "session_id": "sess-42"},
        AuditLog(path=path),
        actor="ci-gate",
    )
    assert load(path)[0]["actor"] == "ci-gate"


def test_a_malformed_event_denial_is_on_the_record(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    res, _ = _run("not json", AuditLog(path=path))
    assert res["hookSpecificOutput"]["permissionDecision"] == "deny"
    entries = load(path)
    assert entries[0]["action"]["tool"] is None
    assert "recusal_malformed_event" in json.dumps(entries[0]["failures"])


def test_a_policy_error_denial_is_on_the_record(tmp_path):
    path = str(tmp_path / "audit.jsonl")

    def _buggy(tool_name, tool_input):
        raise RuntimeError("boom")

    res, _ = _run({"tool_name": "Read", "tool_input": {}}, AuditLog(path=path), policy=_buggy)
    assert res["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "recusal_policy_error" in json.dumps(load(path)[0]["failures"])


def test_an_unwritable_log_fails_closed_to_a_deny(tmp_path):
    class _BrokenLog(AuditLog):
        def append(self, *a, **k):
            raise OSError("disk full")

    res, text = _run(
        {"tool_name": "Read", "tool_input": {}}, _BrokenLog(path=str(tmp_path / "a.jsonl"))
    )
    # the call WOULD have deferred; without the record it must not proceed
    assert res["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "audit log unavailable" in text


def test_an_unwritable_log_with_fail_open_still_defers(tmp_path):
    class _BrokenLog(AuditLog):
        def append(self, *a, **k):
            raise OSError("disk full")

    res, text = _run(
        {"tool_name": "Read", "tool_input": {}},
        _BrokenLog(path=str(tmp_path / "a.jsonl")),
        fail_closed=False,
    )
    assert res is None and text == ""


def test_no_audit_means_no_file_and_identical_behavior(tmp_path):
    res, _ = _run({"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}}, None)
    assert res["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert list(tmp_path.iterdir()) == []


def test_the_chain_grows_across_hook_processes(tmp_path):
    # Each hook invocation is a fresh process; tail resume recovers the head from the
    # final record (no full scan) and the chain stays unbroken across them.
    path = str(tmp_path / "audit.jsonl")
    for i in range(3):
        _run(
            {"tool_name": "Bash", "tool_input": {"command": f"rm -rf /{i}"}},
            AuditLog(path=path, resume="tail"),
        )
    entries = load(path)
    assert [e["seq"] for e in entries] == [0, 1, 2]
    ok, problems = verify_file(path)
    assert ok, problems


@pytest.mark.parametrize("mode", ["full", "tail"])
def test_both_resume_modes_yield_the_same_recorded_chain(tmp_path, mode):
    path = str(tmp_path / f"audit-{mode}.jsonl")
    for _ in range(2):
        _run(
            {"tool_name": "Read", "tool_input": {}},
            AuditLog(path=path, resume=mode),
        )
    ok, problems = verify_file(path)
    assert ok, problems
    assert len(load(path)) == 2


def test_prompt_id_links_the_record_to_the_transcript(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    _run(
        {
            "tool_name": "Read",
            "tool_input": {},
            "session_id": "sess-1",
            "prompt_id": "550e8400-e29b-41d4-a716-446655440000",
        },
        AuditLog(path=path),
    )
    action = load(path)[0]["action"]
    assert action["prompt_id"] == "550e8400-e29b-41d4-a716-446655440000"


def test_a_tool_use_id_is_recorded_defensively_when_present(tmp_path):
    # Not part of the documented PreToolUse event today; recorded if it ever appears.
    path = str(tmp_path / "audit.jsonl")
    _run(
        {"tool_name": "Read", "tool_input": {}, "tool_use_id": "toolu_01abc"},
        AuditLog(path=path),
    )
    assert load(path)[0]["action"]["tool_use_id"] == "toolu_01abc"


def test_the_record_names_the_control_that_decided(tmp_path):
    # A verdict is replayable only when the adjudication rules are identifiable:
    # "same evidence" is insufficient if the policy changed. The package version is
    # automatic; the policy identity is the caller's to declare.
    import recusal

    path = str(tmp_path / "audit.jsonl")
    _run(
        {"tool_name": "Read", "tool_input": {}},
        AuditLog(path=path),
        control={"policy_id": "bank-mcp-write-policy", "policy_version": "3"},
    )
    control = load(path)[0]["action"]["control"]
    assert control["recusal_version"] == recusal.__version__
    assert control["policy_id"] == "bank-mcp-write-policy"
    assert control["policy_version"] == "3"


def test_a_manifest_policy_contributes_its_manifest_digest(tmp_path):
    from recusal.mcp import build_manifest, manifest_policy, manifest_to_text

    manifest_path = tmp_path / "mcp-manifest.json"
    manifest_path.write_text(
        manifest_to_text(build_manifest({"github": [{"name": "create_issue"}]})),
        encoding="utf-8",
    )
    policy = manifest_policy(str(manifest_path))
    path = str(tmp_path / "audit.jsonl")
    _run(
        {"tool_name": "mcp__github__create_issue", "tool_input": {}},
        AuditLog(path=path),
        policy=policy,
    )
    control = load(path)[0]["action"]["control"]
    assert control["manifest_sha256"].startswith("sha256:")
    assert len(control["manifest_sha256"]) == len("sha256:") + 64
