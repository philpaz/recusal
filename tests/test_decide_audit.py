"""``decide(audit=..., surface=...)``: any runtime's adjudication on the record, honestly labeled.

The contract: without ``audit`` nothing changes; with it, every adjudication (defer,
allow, deny) appends one hash-chained entry through the SAME recorder
``run_pretooluse_hook`` uses, naming the runtime that decided in ``surface`` (required,
no default, the hook's own label reserved); the decision and reason are identical to the
unaudited call; tool_input is fingerprinted, never embedded; and an unwritable log
fails CLOSED to a deny unless fail_closed=False.
"""

import io
import itertools
import json

import pytest

import recusal
from recusal import AuditLog, Finding
from recusal.audit import load, verify_file
from recusal.claude_code import decide, run_pretooluse_hook

SURFACE = "my_runtime.tool_gate"


def _deny_rm(tool_name, tool_input):
    if tool_name == "Bash" and "rm -rf" in str(tool_input.get("command", "")):
        return [Finding.fail("destructive_bash", severity="CRITICAL", message="refusing rm -rf")]
    return []


def _broken(tool_name, tool_input):
    raise RuntimeError("policy exploded")


def _ambiguous(tool_name, tool_input):
    return [{"check": "x"}]  # no status/passed: strict mode fails closed


class _BrokenLog(AuditLog):
    def append(self, *a, **k):
        raise OSError("disk full")


# --- without audit: unchanged, and recording arguments are never silently ignored ------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"surface": SURFACE},
        {"tool_use_id": "c1"},
        {"actor": "agent"},
        {"control": {"policy_id": "x"}},
    ],
)
def test_recording_arguments_without_audit_are_refused(kwargs):
    with pytest.raises(ValueError, match="require audit="):
        decide("Bash", {"command": "ls"}, _deny_rm, **kwargs)


def test_no_audit_is_the_pure_decision(tmp_path):
    assert decide("Bash", {"command": "rm -rf /"}, _deny_rm)[0] == "deny"
    assert decide("Bash", {"command": "ls"}, _deny_rm)[0] == "defer"
    assert list(tmp_path.iterdir()) == []


# --- surface is required, honest, and the hook's label is reserved -----------------------


@pytest.mark.parametrize("surface", [None, "", " padded", "padded ", 7])
def test_audit_requires_a_named_surface(tmp_path, surface):
    log = AuditLog(path=str(tmp_path / "a.jsonl"))
    with pytest.raises(ValueError, match="requires surface"):
        decide("Bash", {"command": "ls"}, _deny_rm, audit=log, surface=surface)
    assert not (tmp_path / "a.jsonl").exists()  # refused before anything is written


def test_the_hook_surface_is_reserved(tmp_path):
    log = AuditLog(path=str(tmp_path / "a.jsonl"))
    with pytest.raises(ValueError, match="reserved for run_pretooluse_hook"):
        decide("Bash", {"command": "ls"}, _deny_rm, audit=log, surface="claude_code.pretooluse")


@pytest.mark.parametrize("tool_use_id", ["", 5])
def test_a_tool_use_id_must_be_a_nonempty_string(tmp_path, tool_use_id):
    log = AuditLog(path=str(tmp_path / "a.jsonl"))
    with pytest.raises(ValueError, match="tool_use_id"):
        decide(
            "Bash", {"command": "ls"}, _deny_rm, audit=log, surface=SURFACE, tool_use_id=tool_use_id
        )


# --- with audit: every decision on the record, labeled with the runtime that decided -----


def test_every_decision_is_recorded_once_with_its_surface(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    log = AuditLog(path=path)
    calls = [
        ("Bash", {"command": "rm -rf /"}, "deny", {}),
        ("Bash", {"command": "ls"}, "defer", {}),
        ("Bash", {"command": "ls"}, "allow", {"allow_on_pass": True}),
    ]
    for i, (name, args, expected, kw) in enumerate(calls):
        got = decide(
            name, args, _deny_rm, audit=log, surface=SURFACE, tool_use_id=f"c{i}", actor="a1", **kw
        )
        assert got[0] == expected
    entries = load(path)
    assert [e["action"]["decision"] for e in entries] == ["deny", "defer", "allow"]
    assert [e["action"]["tool_use_id"] for e in entries] == ["c0", "c1", "c2"]
    assert {e["action"]["surface"] for e in entries} == {SURFACE}
    assert {e["actor"] for e in entries} == {"a1"}
    assert entries[0]["decision"] == "FAIL" and entries[1]["decision"] == "PASS"
    assert all(len(e["action"]["input_sha256"]) == 64 for e in entries)
    ok, problems = verify_file(path)
    assert ok, problems


def test_no_tool_use_id_means_no_field(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    decide("Bash", {"command": "ls"}, _deny_rm, audit=AuditLog(path=path), surface=SURFACE)
    assert "tool_use_id" not in load(path)[0]["action"]


def test_the_entry_has_the_same_shape_as_the_hooks(tmp_path):
    # one recorder: identical action keys, in the same order, apart from the label and
    # the hook-only envelope fields (prompt_id, runtime) this call does not carry
    hook_path, decide_path = str(tmp_path / "hook.jsonl"), str(tmp_path / "decide.jsonl")
    event = {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}, "tool_use_id": "c1"}
    run_pretooluse_hook(
        _deny_rm,
        audit=AuditLog(path=hook_path),
        control={"policy_id": "p"},
        stdin=io.StringIO(json.dumps(event)),
        stdout=io.StringIO(),
    )
    decide(
        "Bash",
        {"command": "rm -rf /"},
        _deny_rm,
        audit=AuditLog(path=decide_path),
        surface=SURFACE,
        tool_use_id="c1",
        control={"policy_id": "p"},
    )
    hook_action, decide_action = load(hook_path)[0]["action"], load(decide_path)[0]["action"]
    assert hook_action["surface"] == "claude_code.pretooluse"
    assert decide_action["surface"] == SURFACE
    assert list(hook_action) == list(decide_action)
    assert {k: v for k, v in hook_action.items() if k != "surface"} == {
        k: v for k, v in decide_action.items() if k != "surface"
    }


def test_tool_input_is_fingerprinted_never_embedded(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    secret = "AKIA-SUPER-SECRET-VALUE"
    decide(
        "Write",
        {"file_path": "x", "content": secret},
        _deny_rm,
        audit=AuditLog(path=path),
        surface=SURFACE,
    )
    assert secret not in (tmp_path / "audit.jsonl").read_text(encoding="utf-8")


# --- the decision never changes because it is being recorded -----------------------------

POLICIES = [_deny_rm, _broken, _ambiguous, lambda n, i: None]
NAMES = ["Bash", "Write", "run_sql", "été"]
INPUTS = [
    {},
    {"command": "rm -rf /"},
    {"command": "ls"},
    {"command": None},
    {"command": ["rm", "-rf", "/"]},
    {"sql": "DELETE FROM t"},
    {"nested": {"deep": [1, {"command": "rm -rf /"}]}},
    {"command": "x" * 20000},
]


@pytest.mark.parametrize("kw", [{}, {"allow_on_pass": True}, {"fail_closed": False}])
def test_recording_never_changes_the_decision_or_reason(tmp_path, kw):
    path = str(tmp_path / "audit.jsonl")
    log = AuditLog(path=path)
    n = 0
    for policy, name, args in itertools.product(POLICIES, NAMES, INPUTS):
        assert decide(name, args, policy, audit=log, surface=SURFACE, **kw) == decide(
            name, args, policy, **kw
        )
        n += 1
    entries = load(path)
    assert len(entries) == n
    assert [e["seq"] for e in entries] == list(range(n))
    ok, problems = verify_file(path)
    assert ok, problems


# --- failure paths ----------------------------------------------------------------------


def test_an_unwritable_log_fails_closed_to_a_deny(tmp_path):
    decision, reason = decide(
        "Bash",
        {"command": "ls"},
        _deny_rm,
        audit=_BrokenLog(path=str(tmp_path / "a")),
        surface=SURFACE,
    )
    # the call WOULD have deferred; without the record it must not proceed
    assert decision == "deny"
    assert "audit log unavailable" in reason and "disk full" in reason


def test_an_unwritable_log_with_fail_open_keeps_the_decision(tmp_path):
    log = _BrokenLog(path=str(tmp_path / "a"))
    assert decide(
        "Bash", {"command": "ls"}, _deny_rm, audit=log, surface=SURFACE, fail_closed=False
    ) == decide("Bash", {"command": "ls"}, _deny_rm, fail_closed=False)
    assert (
        decide(
            "Bash", {"command": "rm -rf /"}, _deny_rm, audit=log, surface=SURFACE, fail_closed=False
        )[0]
        == "deny"
    )  # an unwritable log never turns a refusal into a pass


def test_a_policy_error_is_denied_and_recorded(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    decision, reason = decide("Bash", {}, _broken, audit=AuditLog(path=path), surface=SURFACE)
    assert decision == "deny" and "policy exploded" in reason
    entry = load(path)[0]
    assert entry["action"]["decision"] == "deny"
    assert "recusal_policy_error" in json.dumps(entry["failures"])


def test_a_tampered_entry_breaks_the_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path=str(path))
    decide("Bash", {"command": "rm -rf /"}, _deny_rm, audit=log, surface=SURFACE)
    decide("Bash", {"command": "ls"}, _deny_rm, audit=log, surface=SURFACE)
    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["action"]["decision"] = "allow"  # rewrite the refusal
    path.write_text("\n".join([json.dumps(first)] + lines[1:]) + "\n", encoding="utf-8")
    ok, problems = verify_file(str(path))
    assert not ok and problems


# --- control identity: the same authoritative construction as the hook -----------------


def test_caller_cannot_spoof_the_recusal_version(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    decide(
        "Bash",
        {"command": "ls"},
        _deny_rm,
        audit=AuditLog(path=path),
        surface=SURFACE,
        control={"recusal_version": "9.9.9", "policy_id": "x"},
    )
    control = load(path)[0]["action"]["control"]
    assert control["recusal_version"] == recusal.__version__
    assert control["policy_id"] == "x"


def test_a_non_mcp_call_does_not_inherit_stale_manifest_provenance(tmp_path):
    from recusal.mcp import build_manifest, manifest_policy, manifest_to_text

    manifest_path = tmp_path / "mcp-manifest.json"
    manifest_path.write_text(
        manifest_to_text(build_manifest({"github": [{"name": "create_issue"}]})),
        encoding="utf-8",
    )
    policy = manifest_policy(str(manifest_path))
    path = str(tmp_path / "audit.jsonl")
    log = AuditLog(path=path)
    decide("mcp__github__create_issue", {}, policy, audit=log, surface=SURFACE)
    decide("Read", {"file_path": "x"}, policy, audit=log, surface=SURFACE)
    entries = load(path)
    assert entries[0]["action"]["control"]["manifest_sha256"].startswith("sha256:")
    assert "manifest_sha256" not in entries[1]["action"]["control"]  # invocation-local


def test_a_policy_with_a_reset_hook_is_reset_before_each_recorded_call(tmp_path):
    class _Policy:
        resets = 0

        def reset_control_identity(self):
            _Policy.resets += 1

        def __call__(self, tool_name, tool_input):
            return []

    policy = _Policy()
    log = AuditLog(path=str(tmp_path / "a.jsonl"))
    decide("Read", {}, policy, audit=log, surface=SURFACE)
    decide("Read", {}, policy, audit=log, surface=SURFACE)
    assert _Policy.resets == 2


@pytest.mark.parametrize("control", [["policy_id"], "policy_id", 1])
def test_a_non_dict_control_is_refused(tmp_path, control):
    log = AuditLog(path=str(tmp_path / "a.jsonl"))
    with pytest.raises(ValueError, match="control must be a dict"):
        decide("Read", {}, _deny_rm, audit=log, surface=SURFACE, control=control)


def test_an_input_the_record_cannot_bind_fails_closed_and_is_recorded(tmp_path):
    looped = {}
    looped["self"] = looped  # json cannot serialize a cycle, so there is no fingerprint
    path = str(tmp_path / "audit.jsonl")
    decision, reason = decide("Read", looped, _deny_rm, audit=AuditLog(path=path), surface=SURFACE)
    assert decision == "deny" and "cannot be fingerprinted" in reason
    entry = load(path)[0]
    assert entry["action"]["decision"] == "deny"
    assert "input_sha256" not in entry["action"]
    assert "recusal_unfingerprintable_input" in json.dumps(entry["failures"])


def test_an_input_the_record_cannot_bind_with_fail_open_keeps_the_decision(tmp_path):
    looped = {}
    looped["self"] = looped
    path = str(tmp_path / "audit.jsonl")
    got = decide(
        "Read", looped, _deny_rm, audit=AuditLog(path=path), surface=SURFACE, fail_closed=False
    )
    assert got == decide("Read", looped, _deny_rm, fail_closed=False)
    assert "input_sha256" not in load(path)[0]["action"]
