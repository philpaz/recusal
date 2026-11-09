"""``safe_errors=True``: an unexpected exception's text reaches neither the model nor the record.

An exception raised inside a policy, a fingerprint or an audit sink carries whatever its
raiser put in it: a connection string, a credential, a row. By default that text is in the
reason the caller returns to the model and in the finding the audit record stores, as it
always was (the plaintext boundary ``recusal.audit`` documents). ``safe_errors=True``
replaces it with a stable code and the exception's class name.

Every case here plants one sentinel in an exception and holds three things:

- the control: with the option off, the sentinel IS in the reason (so the test would see a
  leak if there were one, and the published strings have not moved);
- with the option on, the sentinel is in neither the returned reason, the hook's output,
  nor one byte of the audit file;
- the decision, the severity and the fail-open behavior are the same in both modes.
"""

import inspect
import io
import json

import pytest

import recusal._adjudicate as adjudicate
from recusal import AuditLog, Finding
from recusal.audit import load, verify_file
from recusal.claude import gate_tool_use, tool_confirmation
from recusal.claude_code import decide, run_pretooluse_hook

SENTINEL = "postgres://svc:PRIVATE_CREDENTIAL_SENTINEL@db.internal/prod"
SURFACE = "my_runtime.tool_gate"


class VendorError(RuntimeError):
    """Stands for an SDK or database exception whose text the application did not write."""


def _raising(tool_name, tool_input):
    raise VendorError(f"connection failed: {SENTINEL}")


def _returns_a_secret_string(tool_name, tool_input):
    return SENTINEL  # not a list of findings: the evaluator raises, quoting the value


def _passing(tool_name, tool_input):
    return []


def _refusing(tool_name, tool_input):
    return [Finding.fail("rule", severity="CRITICAL", message="refused by rule 7")]


class _SinkThatFails:
    def write(self, entry):
        raise VendorError(f"mirror unreachable: {SENTINEL}")


def _file_bytes(path):
    return path.read_text(encoding="utf-8") if path.exists() else ""


# --- the policy call --------------------------------------------------------------------------


@pytest.mark.parametrize("policy", [_raising, _returns_a_secret_string])
def test_a_policy_exception_is_a_code_in_the_reason(policy):
    legacy = decide("run_sql", {"sql": "SELECT 1"}, policy)
    assert legacy[0] == "deny" and SENTINEL in legacy[1]  # the control: unchanged by default

    decision, reason = decide("run_sql", {"sql": "SELECT 1"}, policy, safe_errors=True)
    assert decision == "deny"
    assert SENTINEL not in reason and "PRIVATE" not in reason
    expected = "VendorError" if policy is _raising else "TypeError"
    assert reason == f"Recusal failed closed (policy error): RECUSAL_POLICY_ERROR ({expected})"


def test_a_policy_exception_is_a_code_on_the_record(tmp_path):
    legacy_path, safe_path = tmp_path / "legacy.jsonl", tmp_path / "safe.jsonl"
    decide("run_sql", {}, _raising, audit=AuditLog(path=str(legacy_path)), surface=SURFACE)
    assert SENTINEL in _file_bytes(legacy_path)  # the control

    decision, reason = decide(
        "run_sql",
        {},
        _raising,
        audit=AuditLog(path=str(safe_path)),
        surface=SURFACE,
        safe_errors=True,
    )
    assert decision == "deny" and SENTINEL not in reason
    assert "PRIVATE" not in _file_bytes(safe_path)
    safe, legacy = load(str(safe_path))[0], load(str(legacy_path))[0]
    assert safe["action"]["reason"] == reason
    assert [f["message"] for f in safe["failures"]] == ["RECUSAL_POLICY_ERROR (VendorError)"]
    # the same verdict, the same severity, the same finding: only the text differs
    assert safe["decision"] == legacy["decision"] == "FAIL"
    assert [(f["check"], f["severity"]) for f in safe["failures"]] == [
        (f["check"], f["severity"]) for f in legacy["failures"]
    ]
    ok, problems = verify_file(str(safe_path))
    assert ok, problems


@pytest.mark.parametrize("allow_on_pass", [False, True])
def test_fail_open_keeps_its_decision_and_loses_the_text(tmp_path, allow_on_pass):
    kw = {"fail_closed": False, "allow_on_pass": allow_on_pass}
    legacy = decide("run_sql", {}, _raising, **kw)
    assert SENTINEL in legacy[1]

    path = tmp_path / "audit.jsonl"
    safe = decide(
        "run_sql",
        {},
        _raising,
        audit=AuditLog(path=str(path)),
        surface=SURFACE,
        safe_errors=True,
        **kw,
    )
    assert safe[0] == legacy[0] == ("allow" if allow_on_pass else "defer")
    assert safe[1] == "policy error ignored: RECUSAL_POLICY_ERROR (VendorError)"
    assert "PRIVATE" not in _file_bytes(path)
    entry = load(str(path))[0]
    assert entry["decision"] == "PASS"  # WARNING, as in the default mode
    assert entry["highest_severity"] == "WARNING" and entry["action"]["reason"] == safe[1]
    # the synthesized warning itself (a PASS record stores no failures, so read the verdict)
    _, _, verdict = adjudicate._adjudicate(
        "run_sql",
        {},
        _raising,
        fail_closed=False,
        safe_errors=True,
        **{"allow_on_pass": allow_on_pass},
    )
    assert [w.message for w in verdict.warnings] == [
        "ignored (fail_closed=False): RECUSAL_POLICY_ERROR (VendorError)"
    ]


# --- the fingerprint that binds the input to its record --------------------------------------


def test_an_input_binding_exception_is_a_code(tmp_path, monkeypatch):
    def cannot_fingerprint(tool_input):
        raise VendorError(f"cannot serialize {SENTINEL}")

    monkeypatch.setattr(adjudicate, "_input_fingerprint", cannot_fingerprint)
    legacy_path, safe_path = tmp_path / "legacy.jsonl", tmp_path / "safe.jsonl"
    legacy = decide("Read", {}, _passing, audit=AuditLog(path=str(legacy_path)), surface=SURFACE)
    assert legacy[0] == "deny" and SENTINEL in legacy[1] and SENTINEL in _file_bytes(legacy_path)

    called = []

    def recording(tool_name, tool_input):
        called.append(tool_name)
        return []

    decision, reason = decide(
        "Read",
        {},
        recording,
        audit=AuditLog(path=str(safe_path)),
        surface=SURFACE,
        safe_errors=True,
    )
    assert decision == "deny" and called == []
    assert reason == (
        "Recusal failed closed: tool_input cannot be fingerprinted "
        "(RECUSAL_INPUT_BINDING_ERROR (VendorError))"
    )
    assert "PRIVATE" not in _file_bytes(safe_path)
    assert [f["message"] for f in load(str(safe_path))[0]["failures"]] == [
        "RECUSAL_INPUT_BINDING_ERROR (VendorError)"
    ]


def test_a_real_unfingerprintable_input_names_its_class_only(tmp_path):
    class Opaque:
        def __repr__(self):
            return SENTINEL

    path = tmp_path / "audit.jsonl"
    decision, reason = decide(
        "Read",
        {"handle": Opaque()},
        _passing,
        audit=AuditLog(path=str(path)),
        surface=SURFACE,
        safe_errors=True,
    )
    assert decision == "deny" and "RECUSAL_INPUT_BINDING_ERROR" in reason
    assert "PRIVATE" not in reason and "PRIVATE" not in _file_bytes(path)


# --- the audit write ---------------------------------------------------------------------------


def test_an_audit_sink_exception_is_a_code(tmp_path):
    def log(name):
        return AuditLog(path=str(tmp_path / name), sinks=[_SinkThatFails()])

    legacy = decide("Read", {}, _passing, audit=log("legacy.jsonl"), surface=SURFACE)
    assert legacy[0] == "deny" and SENTINEL in legacy[1]

    decision, reason = decide(
        "Read", {}, _passing, audit=log("safe.jsonl"), surface=SURFACE, safe_errors=True
    )
    assert decision == "deny"  # the call would have deferred; without the record it does not run
    assert "PRIVATE" not in reason
    assert reason.startswith(
        "Recusal failed closed: audit log unavailable (RECUSAL_AUDIT_UNAVAILABLE ("
    )
    assert reason.endswith("); the record is part of the control")
    assert "PRIVATE" not in _file_bytes(tmp_path / "safe.jsonl")


def test_an_audit_failure_with_fail_open_keeps_the_decision_in_both_modes(tmp_path):
    def log(name):
        return AuditLog(path=str(tmp_path / name), sinks=[_SinkThatFails()])

    for policy in (_passing, _refusing):
        legacy = decide(
            "Read", {}, policy, audit=log("a.jsonl"), surface=SURFACE, fail_closed=False
        )
        safe = decide(
            "Read",
            {},
            policy,
            audit=log("b.jsonl"),
            surface=SURFACE,
            fail_closed=False,
            safe_errors=True,
        )
        assert safe == legacy == decide("Read", {}, policy, fail_closed=False)


# --- the hook ----------------------------------------------------------------------------------


def _hook(event, policy, **kwargs):
    out = io.StringIO()
    stdin = io.StringIO(event if isinstance(event, str) else json.dumps(event))
    result = run_pretooluse_hook(policy, stdin=stdin, stdout=out, **kwargs)
    return result, out.getvalue()


def test_the_hook_emits_a_code_for_a_policy_exception(tmp_path):
    event = {"tool_name": "Bash", "tool_input": {"command": "ls"}}
    _, legacy_text = _hook(event, _raising)
    assert SENTINEL in legacy_text  # the control

    path = tmp_path / "audit.jsonl"
    result, text = _hook(event, _raising, audit=AuditLog(path=str(path)), safe_errors=True)
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert result["hookSpecificOutput"]["permissionDecisionReason"] == (
        "Recusal failed closed (policy error): RECUSAL_POLICY_ERROR (VendorError)"
    )
    assert "PRIVATE" not in text and "PRIVATE" not in _file_bytes(path)


def test_the_hook_emits_a_code_for_a_malformed_event(tmp_path):
    class Stdin:
        def read(self, *args):
            raise VendorError(f"transport said {SENTINEL}")

    def run(**kwargs):
        out = io.StringIO()
        result = run_pretooluse_hook(_passing, stdin=Stdin(), stdout=out, **kwargs)
        return result, out.getvalue()

    _, legacy_text = run()
    assert SENTINEL in legacy_text  # the control

    path = tmp_path / "audit.jsonl"
    result, text = run(audit=AuditLog(path=str(path)), safe_errors=True)
    assert result["hookSpecificOutput"]["permissionDecisionReason"] == (
        "Recusal failed closed: malformed PreToolUse event (RECUSAL_MALFORMED_EVENT (VendorError))"
    )
    assert "PRIVATE" not in text and "PRIVATE" not in _file_bytes(path)
    assert [f["message"] for f in load(str(path))[0]["failures"]] == [
        "RECUSAL_MALFORMED_EVENT (VendorError)"
    ]

    # fail-open defers in both modes, and its record carries the code
    open_path = tmp_path / "open.jsonl"
    result, text = run(audit=AuditLog(path=str(open_path)), safe_errors=True, fail_closed=False)
    assert result is None and text == ""
    assert "PRIVATE" not in _file_bytes(open_path)
    assert load(str(open_path))[0]["action"]["reason"] == (
        "malformed PreToolUse event ignored: RECUSAL_MALFORMED_EVENT (VendorError)"
    )


def test_an_ordinary_malformed_event_keeps_its_text_by_default():
    result, _ = _hook("not json", _passing)
    reason = result["hookSpecificOutput"]["permissionDecisionReason"]
    assert reason.startswith("Recusal failed closed: malformed PreToolUse event (Expecting value")
    result, _ = _hook("not json", _passing, safe_errors=True)
    assert result["hookSpecificOutput"]["permissionDecisionReason"] == (
        "Recusal failed closed: malformed PreToolUse event "
        "(RECUSAL_MALFORMED_EVENT (JSONDecodeError))"
    )


# --- the evidence helpers ----------------------------------------------------------------------


class _EvidenceThatFails:
    def __iter__(self):
        raise VendorError(f"evidence store said {SENTINEL}")


def test_gate_tool_use_names_a_code_for_evidence_it_cannot_fold():
    allow, block = gate_tool_use("t1", _EvidenceThatFails(), tool_name="run_sql")
    assert not allow and SENTINEL in block["content"]  # the control

    allow, block = gate_tool_use("t1", _EvidenceThatFails(), tool_name="run_sql", safe_errors=True)
    assert not allow and block["is_error"] is True
    assert block["content"] == (
        "Refused by the gate before `run_sql` ran invalid evidence "
        "(RECUSAL_INVALID_EVIDENCE (VendorError)). "
        "Address the findings and propose a corrected call."
    )


def test_tool_confirmation_names_a_code_for_evidence_it_cannot_fold():
    assert SENTINEL in tool_confirmation("t1", _EvidenceThatFails())["deny_message"]
    event = tool_confirmation("t1", _EvidenceThatFails(), safe_errors=True)
    assert event["result"] == "deny"
    assert event["deny_message"] == "invalid evidence (RECUSAL_INVALID_EVIDENCE (VendorError))"


# --- what the option does not touch ------------------------------------------------------------


def test_a_policys_own_finding_messages_are_never_rewritten(tmp_path):
    path = tmp_path / "audit.jsonl"
    legacy = decide("run_sql", {}, _refusing)
    safe = decide(
        "run_sql", {}, _refusing, audit=AuditLog(path=str(path)), surface=SURFACE, safe_errors=True
    )
    assert safe == legacy and "refused by rule 7" in safe[1]
    assert "refused by rule 7" in _file_bytes(path)


@pytest.mark.parametrize("policy", [_passing, _refusing])
@pytest.mark.parametrize("allow_on_pass", [False, True])
def test_a_decision_without_an_exception_is_identical_in_both_modes(policy, allow_on_pass):
    kw = {"allow_on_pass": allow_on_pass}
    assert decide("run_sql", {}, policy, safe_errors=True, **kw) == decide(
        "run_sql", {}, policy, **kw
    )


def test_the_option_is_keyword_only_and_off_by_default_on_every_entry_point():
    for func in (decide, run_pretooluse_hook, gate_tool_use, tool_confirmation):
        parameter = inspect.signature(func).parameters["safe_errors"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, func.__name__
        assert parameter.default is False, func.__name__


def test_the_codes_are_the_published_ones():
    assert (
        adjudicate._POLICY_ERROR,
        adjudicate._INPUT_BINDING_ERROR,
        adjudicate._AUDIT_UNAVAILABLE,
        adjudicate._MALFORMED_EVENT,
        adjudicate._INVALID_EVIDENCE,
    ) == (
        "RECUSAL_POLICY_ERROR",
        "RECUSAL_INPUT_BINDING_ERROR",
        "RECUSAL_AUDIT_UNAVAILABLE",
        "RECUSAL_MALFORMED_EVENT",
        "RECUSAL_INVALID_EVIDENCE",
    )


def test_no_exception_text_site_in_the_decision_boundary_bypasses_the_formatter():
    """A new ``except`` that formats its exception into a reason must go through
    ``_error_detail``; this fails when one is added that does not."""
    import recusal.claude as claude
    import recusal.claude_code as claude_code

    for module in (adjudicate, claude):
        source = inspect.getsource(module)
        assert "{exc}" not in source and "str(exc)" not in source.replace("else str(exc)", ""), (
            module.__name__
        )
    hook = inspect.getsource(claude_code.run_pretooluse_hook)
    assert "{exc}" not in hook and "str(exc)" not in hook
