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
import threading

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


# ------------------------------------------------- one copy per holder
#
# Every holder of an entry other than the log (each sink, the caller of append) has a
# dict of its own. A sink that changes its argument, at once or later, and a caller that
# changes what append returned, reach no other sink, not the retained mirror, not the
# head and not the returned entry. The three logs under test: in memory, file-backed
# with the mirror retained (resume="full"), file-backed without it (resume="tail").

MODES = ("memory", "full", "tail")


class _Opaque:
    def __str__(self):
        return "opaque-object"


def _action(n):
    """Everything an action can hold that the stored form changes or has to carry."""
    return {
        "tool": "bash",
        "n": n,
        "rows": [{"k": "v"}, {"k": "w"}],
        "codes": {1: "a", 2: "b", 10: "c"},  # integer keys: strings once stored
        "pair": (1, 2),  # a tuple: a list once stored
        "thing": _Opaque(),  # not JSON: its text once stored
        "big": 2**70,
        "ratio": 0.1,
        "unbounded": float("inf"),
        "text": "café \U0001f600",
    }


def _stamp(n):
    return f"2026-01-01T00:00:{n:02d}+00:00"


def _append(log, n):
    return log.append(_verdict(n % 2 == 0), action=_action(n), actor="agent-1", timestamp=_stamp(n))


def _reference(tmp_path, count):
    """The chain those appends make, written by a log with no sinks and read back from
    its file: no object in it was ever handed to a sink or returned to a caller."""
    path = str(tmp_path / "reference.jsonl")
    log = AuditLog(path=path, resume="tail")
    for n in range(count):
        _append(log, n)
    entries = load(path)
    ok, problems = verify(entries)
    assert ok and len(entries) == count, problems
    return entries


def _open(mode, tmp_path, sinks):
    if mode == "memory":
        return AuditLog(sinks=sinks), None
    path = str(tmp_path / "audit.jsonl")
    return AuditLog(path=path, resume=mode, sinks=sinks), path


def _assert_intact(mode, log, path, reference, records=()):
    """The file, the retained mirror, the head and each given record are the reference
    chain, and each verifies on its own."""
    assert log.last_hash == reference[-1]["hash"]
    if path is not None:
        ok, problems = verify_file(path, expected_head=(len(reference), log.last_hash))
        assert ok, problems
        assert load(path) == reference
    if mode == "tail":
        assert log.entries == []
    else:
        assert log.entries == reference
        ok, problems = verify(log.entries)
        assert ok, problems
    for record in records:
        assert record == reference
        ok, problems = verify(record)
        assert ok, problems
        if path is not None:
            # byte for byte the lines of the file, so a sink that writes what it is
            # handed (examples/audit_sink.py) mirrors the file exactly
            lines = open(path, encoding="utf-8").read().splitlines()
            assert [json.dumps(entry, ensure_ascii=True) for entry in record] == lines


def _change_a_top_level_field(entry):
    entry["actor"] = "someone-else"


def _change_a_nested_field(entry):
    entry["action"]["tool"] = entry["action"]["tool"].upper()


def _change_inside_a_nested_list(entry):
    entry["action"]["rows"][0]["k"] = "changed"
    entry["failures"].append({"check": "added by a sink"})


def _replace_a_nested_container(entry):
    entry["action"] = {"tool": "something else"}
    entry["failures"] = []


def _rewrite_the_identity(entry):
    entry["seq"] = 99
    entry["prev_hash"] = "e" * 64
    entry["hash"] = "f" * 64


def _empty_it(entry):
    entry["action"].clear()
    entry.clear()


CHANGES = (
    _change_a_top_level_field,
    _change_a_nested_field,
    _change_inside_a_nested_list,
    _replace_a_nested_container,
    _rewrite_the_identity,
    _empty_it,
)


class ChangingSink:
    """Changes its argument without raising, as a sink that formats in place would."""

    def __init__(self, change):
        self.change = change

    def write(self, entry):
        self.change(entry)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("change", CHANGES, ids=lambda change: change.__name__.strip("_"))
def test_a_sink_that_changes_its_entry_reaches_nothing_else(tmp_path, mode, change):
    reference = _reference(tmp_path, 3)
    before, after = RecorderSink(), RecorderSink()
    log, path = _open(mode, tmp_path, [before, ChangingSink(change), after])
    returned = [_append(log, n) for n in range(3)]
    _assert_intact(mode, log, path, reference, [before.entries, after.entries, returned])


@pytest.mark.parametrize("mode", MODES)
def test_every_holder_of_an_entry_has_its_own_objects(tmp_path, mode):
    a, b = RecorderSink(), RecorderSink()
    log, path = _open(mode, tmp_path, [a, b])
    returned = _append(log, 0)
    holders = [a.entries[0], b.entries[0], returned] + log.entries
    assert len(holders) == (3 if mode == "tail" else 4)
    for i, one in enumerate(holders):
        for other in holders[i + 1 :]:
            assert one == other
            assert one is not other
            assert one["action"] is not other["action"]
            assert one["action"]["rows"] is not other["action"]["rows"]
            assert one["action"]["rows"][0] is not other["action"]["rows"][0]
            assert one["action"]["codes"] is not other["action"]["codes"]
            assert one["failures"] is not other["failures"]


@pytest.mark.parametrize("mode", MODES)
def test_a_sink_is_handed_the_stored_form_with_the_same_seq_and_hash(tmp_path, mode):
    sink = RecorderSink()
    log, path = _open(mode, tmp_path, [ChangingSink(_empty_it), sink])
    returned = _append(log, 0)
    (seen,) = sink.entries
    assert seen["action"]["codes"] == {"1": "a", "2": "b", "10": "c"}
    assert seen["action"]["pair"] == [1, 2]
    assert seen["action"]["thing"] == "opaque-object"
    assert seen["action"]["big"] == 2**70 and seen["action"]["unbounded"] == float("inf")
    assert (seen["seq"], seen["hash"]) == (0, log.last_hash) == (returned["seq"], returned["hash"])
    assert list(seen) == list(returned)  # the documented fields, in the order written
    assert verify([seen]) == (True, [])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("change", CHANGES, ids=lambda change: change.__name__.strip("_"))
def test_changing_the_returned_entry_reaches_nothing_else(tmp_path, mode, change):
    reference = _reference(tmp_path, 3)
    sink = RecorderSink()
    log, path = _open(mode, tmp_path, [sink])
    for n in range(3):
        change(_append(log, n))  # the caller changes what append returned, every time
    _assert_intact(mode, log, path, reference, [sink.entries])


@pytest.mark.parametrize("mode", ("memory", "full"))
def test_changing_the_returned_entry_leaves_the_retained_chain_intact_without_sinks(tmp_path, mode):
    reference = _reference(tmp_path, 3)
    log, path = _open(mode, tmp_path, None)
    for n in range(3):
        returned = _append(log, n)
        assert returned == reference[n] and returned is not log.entries[n]
        _empty_it(returned)
    _assert_intact(mode, log, path, reference)


@pytest.mark.parametrize("mode", MODES)
def test_a_sink_that_keeps_its_entry_and_changes_it_later_reaches_nothing_else(tmp_path, mode):
    reference = _reference(tmp_path, 4)
    keeper, other = RecorderSink(), RecorderSink()
    log, path = _open(mode, tmp_path, [keeper, other])
    returned = [_append(log, 0), _append(log, 1)]
    for kept, change in zip(keeper.entries, (_change_a_nested_field, _rewrite_the_identity)):
        change(kept)  # after append returned
    returned.append(_append(log, 2))
    _empty_it(keeper.entries[2])
    returned.append(_append(log, 3))
    assert returned[2]["prev_hash"] == reference[1]["hash"]  # chained to the true head
    _assert_intact(mode, log, path, reference, [other.entries, returned])


@pytest.mark.parametrize("resume", ("full", "tail"))
def test_a_resumed_log_hands_each_sink_its_own_copy(tmp_path, resume):
    reference = _reference(tmp_path, 4)
    path = str(tmp_path / "audit.jsonl")
    first = AuditLog(path=path)
    for n in range(2):
        _append(first, n)
    sink = RecorderSink()
    log = AuditLog(
        path=path,
        resume=resume,
        sinks=[ChangingSink(_change_a_nested_field), sink, ChangingSink(_rewrite_the_identity)],
        expected_head=(2, reference[1]["hash"]),
    )
    returned = [_append(log, n) for n in (2, 3)]
    assert sink.entries == reference[2:] == returned
    assert load(path) == reference
    assert verify(reference[:2] + sink.entries) == (True, [])
    assert verify_file(path, expected_head=(4, log.last_hash)) == (True, [])
    assert log.entries == (reference if resume == "full" else [])
    assert verify(log.entries) == (True, [])


class ChangeThenFailSink:
    """Changes its argument, then reports failure: the worst of both."""

    def __init__(self, fail_times):
        self.calls = 0
        self.fail_times = fail_times

    def write(self, entry):
        self.calls += 1
        _change_a_nested_field(entry)
        _rewrite_the_identity(entry)
        if self.calls <= self.fail_times:
            raise RuntimeError("mirror store unreachable")


@pytest.mark.parametrize("resume", ("full", "tail"))
def test_a_file_backed_sink_failure_keeps_its_order_and_state(tmp_path, resume):
    """The file is written before any sink is called; a sink that raises stops the sinks
    after it and leaves this process's head and mirror where they were; the next append
    takes the head from the file."""
    reference = _reference(tmp_path, 2)
    before, failing, after = RecorderSink(), ChangeThenFailSink(1), RecorderSink()
    log, path = _open(resume, tmp_path, [before, failing, after])
    with pytest.raises(RuntimeError, match="mirror store unreachable"):
        _append(log, 0)
    assert load(path) == reference[:1]  # on record, as written
    assert before.entries == reference[:1] and after.entries == []
    assert log.entries == [] and log.last_hash == GENESIS  # not advanced in this process
    returned = _append(log, 1)
    assert returned == reference[1] and log.last_hash == reference[1]["hash"]
    assert load(path) == reference and verify_file(path) == (True, [])
    assert before.entries == reference and after.entries == reference[1:]
    assert verify(before.entries) == (True, [])
    # the mirror holds what this process completed, never the entry whose sink failed
    assert log.entries == (reference[1:] if resume == "full" else [])


def test_an_in_memory_sink_failure_keeps_its_order_and_state(tmp_path):
    """An in-memory log has advanced its entry and its head before any sink is called,
    so a sink that raises leaves the entry on record and stops the sinks after it."""
    reference = _reference(tmp_path, 2)
    before, failing, after = RecorderSink(), ChangeThenFailSink(1), RecorderSink()
    log = AuditLog(sinks=[before, failing, after])
    with pytest.raises(RuntimeError, match="mirror store unreachable"):
        _append(log, 0)
    assert log.entries == reference[:1] and log.last_hash == reference[0]["hash"]
    assert before.entries == reference[:1] and after.entries == []
    returned = _append(log, 1)
    assert returned == reference[1]
    assert before.entries == reference and after.entries == reference[1:]
    _assert_intact("memory", log, None, reference, [before.entries])


@pytest.mark.parametrize("mode", MODES)
def test_threads_with_a_changing_sink_still_mirror_one_intact_chain(tmp_path, mode):
    sink = RecorderSink()
    log, path = _open(mode, tmp_path, [ChangingSink(_rewrite_the_identity), sink])
    returned = []

    def work():
        for _ in range(5):
            entry = log.append(_verdict(True), action={"tool": "bash", "rows": [1, 2]})
            returned.append(entry["hash"])
            _empty_it(entry)

    threads = [threading.Thread(target=work) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(sink.entries) == 30
    ok, problems = verify(sink.entries, expected_head=(30, log.last_hash))
    assert ok, problems
    assert sorted(returned) == sorted(entry["hash"] for entry in sink.entries)
    if path is not None:
        assert load(path) == sink.entries
    if mode != "tail":
        assert log.entries == sink.entries


# Actions two containers deep, with the values the stored form changes: integer keys
# and tuples. Built flat (no recursive strategy) so generation stays quick everywhere.
_SCALARS = (
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(max_size=8)
)
_KEYS = st.text(max_size=3) | st.integers(-20, 20)
_FLAT = st.lists(_SCALARS, max_size=3) | st.dictionaries(_KEYS, _SCALARS, max_size=3)
_ACTIONS = st.dictionaries(
    _KEYS,
    _SCALARS | _FLAT | st.lists(_FLAT, max_size=2) | st.tuples(_SCALARS, _FLAT),
    max_size=4,
)


def _scramble(value):
    """Change every container in an entry, at every depth."""
    if isinstance(value, dict):
        for inner in list(value.values()):
            _scramble(inner)
        value.clear()
        value["scrambled"] = True
    elif isinstance(value, list):
        for inner in value:
            _scramble(inner)
        value[:] = ["scrambled"]


@DETERMINISTIC
@given(st.lists(_ACTIONS, min_size=1, max_size=3))
def test_no_action_lets_a_sink_or_a_caller_reach_another_holder(actions):
    def run(sinks, after_append):
        log = AuditLog(sinks=sinks)
        for n, action in enumerate(actions):
            after_append(log.append(_verdict(True), action=action, timestamp=_stamp(n)))
        return log

    untouched = run([], lambda entry: None)
    reference = json.loads(json.dumps(untouched.entries))
    sink = RecorderSink()
    log = run([ChangingSink(_scramble), sink, ChangingSink(_scramble)], _scramble)
    assert log.entries == reference == sink.entries
    assert log.last_hash == reference[-1]["hash"]
    assert verify(log.entries) == (True, []) and verify(sink.entries) == (True, [])
