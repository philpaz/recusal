"""AuditSink mirroring + verify_on_open / expected_head anchoring (ledger item 4).

The chain alone is tamper-evident, not tamper-proof: a writer can rewrite or truncate
the tail. The countermeasure is external - mirror entries to a store the attacker
cannot rewrite (AuditSink) and anchor resumes against a head committed there
(verify_on_open= / expected_head=). The contract under test: every committed entry
reaches every sink in chain order; a sink failure raises out of append (callers fail
closed, the hook denies) while the local record remains; a log that fails strict
verification, or misses its anchor, refuses to open at all.
"""

import io
import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import recusal
from recusal import AuditIntegrityError, AuditLog, AuditSink, Finding, compute_verdict
from recusal.audit import GENESIS, load, verify, verify_file
from recusal.claude_code import run_pretooluse_hook

DETERMINISTIC = settings(derandomize=True, max_examples=60)


def _verdict(passed=True):
    make = Finding.ok if passed else Finding.fail
    return compute_verdict([make("demo", message="observed")])


class RecorderSink:
    def __init__(self):
        self.entries = []

    def write(self, entry):
        self.entries.append(entry)


class HeadSink:
    """The anchor recipe from the AuditSink docstring, verbatim."""

    def __init__(self):
        self.head = (0, GENESIS)

    def write(self, entry):
        self.head = (entry["seq"] + 1, entry["hash"])


class FailingSink:
    def __init__(self, fail_times=None):
        self.calls = 0
        self.fail_times = fail_times

    def write(self, entry):
        self.calls += 1
        if self.fail_times is None or self.calls <= self.fail_times:
            raise RuntimeError("mirror store unreachable")


# ---------------------------------------------------------------- sinks


def test_every_committed_entry_reaches_every_sink_in_chain_order(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    a, b = RecorderSink(), RecorderSink()
    log = AuditLog(path=path, sinks=[a, b])
    log.append(_verdict(True), action={"n": 1})
    log.append(_verdict(False), action={"n": 2})
    log.append(_verdict(True), action={"n": 3})
    on_disk = load(path)
    assert a.entries == on_disk == b.entries
    assert [e["seq"] for e in a.entries] == [0, 1, 2]


def test_an_in_memory_log_notifies_sinks_too():
    sink = RecorderSink()
    log = AuditLog(sinks=(sink,))
    log.append(_verdict(True))
    assert sink.entries == log.entries and len(sink.entries) == 1


def test_sinks_must_be_a_list_or_tuple():
    with pytest.raises(ValueError, match="list or tuple"):
        AuditLog(sinks={RecorderSink()})
    with pytest.raises(ValueError, match="list or tuple"):
        AuditLog(sinks=iter([RecorderSink()]))


def test_a_sink_without_a_callable_write_is_refused_at_construction():
    class NotASink:
        write = "not callable"

    with pytest.raises(ValueError, match="callable write"):
        AuditLog(sinks=[NotASink()])


def test_a_failing_sink_raises_and_the_local_entry_remains(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    log = AuditLog(path=path, sinks=[FailingSink()])
    with pytest.raises(RuntimeError, match="mirror store unreachable"):
        log.append(_verdict(True))
    entries = load(path)
    assert len(entries) == 1  # the local record committed before the sink spoke
    ok, problems = verify_file(path)
    assert ok, problems


def test_the_chain_continues_intact_after_a_sink_failure(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    flaky = FailingSink(fail_times=1)
    log = AuditLog(path=path, sinks=[flaky])
    with pytest.raises(RuntimeError):
        log.append(_verdict(True))
    log.append(_verdict(True))  # the retry is a NEW entry; the head re-derives from file
    entries = load(path)
    assert [e["seq"] for e in entries] == [0, 1]
    ok, problems = verify_file(path)
    assert ok, problems
    assert flaky.calls == 2


def test_a_failing_sink_fails_the_hook_closed_to_a_deny(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    audit = AuditLog(path=path, sinks=[FailingSink()])
    out = io.StringIO()
    stdin = io.StringIO(json.dumps({"tool_name": "Read", "tool_input": {"file_path": "x"}}))
    res = run_pretooluse_hook(lambda name, inp: [], audit=audit, stdin=stdin, stdout=out)
    assert res["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert len(load(path)) == 1  # denied to the agent, yet the adjudication is on record


def test_a_head_recording_sink_anchors_verification(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    sink = HeadSink()
    log = AuditLog(path=path, sinks=[sink])
    for _ in range(3):
        log.append(_verdict(True))
    ok, problems = verify_file(path, expected_head=sink.head)
    assert ok, problems
    # Truncate the tail: the bare chain stays self-consistent, only the anchor objects.
    lines = open(path, encoding="utf-8").read().splitlines()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines[:2]) + "\n")
    assert verify_file(path)[0]
    ok, problems = verify_file(path, expected_head=sink.head)
    assert not ok and any("truncation" in p or "head mismatch" in p for p in problems)


# ---------------------------------------------------- verify_on_open / anchor


def _make_log(path, n=3):
    log = AuditLog(path=path)
    for _ in range(n):
        log.append(_verdict(True))
    return log.last_hash


def test_verify_on_open_resumes_an_intact_log(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    _make_log(path)
    full = AuditLog(path=path, resume="full", verify_on_open=True)
    tail = AuditLog(path=path, resume="tail", verify_on_open=True)
    assert len(full.entries) == 3 and tail.entries == []
    assert full.append(_verdict(True))["seq"] == 3
    ok, problems = verify_file(path)
    assert ok, problems


def test_verify_on_open_refuses_a_tampered_log(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    _make_log(path)
    lines = open(path, encoding="utf-8").read().splitlines()
    doctored = json.loads(lines[0])
    doctored["actor"] = "someone-else"  # in-place edit, hash left as it was
    lines[0] = json.dumps(doctored)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with pytest.raises(AuditIntegrityError) as exc:
        AuditLog(path=path, verify_on_open=True)
    assert exc.value.path == path
    assert any("tampered" in p for p in exc.value.problems)


def test_truncation_needs_the_anchor_to_be_caught(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    last_hash = _make_log(path)
    lines = open(path, encoding="utf-8").read().splitlines()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines[:2]) + "\n")
    # The honest boundary: a truncated tail is still a self-consistent chain.
    AuditLog(path=path, verify_on_open=True)
    with pytest.raises(AuditIntegrityError):
        AuditLog(path=path, expected_head=(3, last_hash))


def test_a_missing_file_is_a_new_log_not_a_failure(tmp_path):
    path = str(tmp_path / "fresh.jsonl")
    log = AuditLog(path=path, verify_on_open=True)
    assert log.append(_verdict(True))["seq"] == 0


def test_a_missing_file_fails_against_a_nonempty_anchor(tmp_path):
    path = str(tmp_path / "gone.jsonl")
    with pytest.raises(AuditIntegrityError, match="missing log is not an intact log"):
        AuditLog(path=path, expected_head=(2, "ab" * 32))


def test_an_empty_anchor_accepts_a_missing_file(tmp_path):
    path = str(tmp_path / "fresh.jsonl")
    log = AuditLog(path=path, expected_head=(0, GENESIS))
    assert log.append(_verdict(True))["seq"] == 0


def test_expected_head_implies_verification(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    _make_log(path)
    with pytest.raises(AuditIntegrityError, match="head mismatch"):
        AuditLog(path=path, expected_head=(3, "cd" * 32))  # verify_on_open never passed


def test_a_correct_anchor_resumes(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    last_hash = _make_log(path)
    log = AuditLog(path=path, expected_head=(3, last_hash), resume="tail")
    assert log.append(_verdict(True))["seq"] == 3


def test_verify_on_open_requires_a_path():
    with pytest.raises(ValueError, match="requires a path"):
        AuditLog(verify_on_open=True)


def test_the_new_names_export_from_the_package_root():
    assert "AuditSink" in recusal.__all__ and "AuditIntegrityError" in recusal.__all__
    assert issubclass(AuditIntegrityError, Exception)
    assert AuditSink is recusal.AuditSink


# ------------------------------------------------------------- property


@DETERMINISTIC
@given(st.lists(st.booleans(), max_size=8))
def test_the_mirror_always_equals_the_log_and_anchors_it(outcomes):
    sink = RecorderSink()
    log = AuditLog(sinks=[sink])
    for passed in outcomes:
        log.append(_verdict(passed))
    assert sink.entries == log.entries
    anchor = (len(log.entries), log.last_hash)
    ok, problems = verify(log.entries, expected_head=anchor)
    assert ok, problems
