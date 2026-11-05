"""The line framing of the audit record: an append never writes onto an unterminated line.

One entry per line is what makes a record readable. Before this guard, ``append`` wrote the
new entry at EOF regardless of what the file ended in: a complete final record whose newline
was lost, or an incomplete record left by a killed writer, fused with the new entry into one
line that no reader could parse, while ``append`` reported success and the gate let the
call proceed. These tests assert the FILE after the append, not the returned entry.
"""

import json

import pytest

from recusal import AuditFramingError, AuditIntegrityError, AuditLog, compute_verdict
from recusal.audit import load, verify, verify_file
from recusal.claude_code import decide

PARTIAL = b'{"seq": 1, "interrupted":'


def _verdict():
    return compute_verdict([])


def _strip_final_newline(path):
    raw = path.read_bytes()
    path.write_bytes(raw.rstrip(b"\r\n"))


def _entries_chained(entries):
    return all(entries[i]["prev_hash"] == entries[i - 1]["hash"] for i in range(1, len(entries)))


# --- a complete final record that lost its newline -------------------------------------------


@pytest.mark.parametrize("resume", ["full", "tail"])
@pytest.mark.parametrize("fsync", [False, True])
def test_a_valid_unterminated_tail_becomes_two_readable_chained_entries(tmp_path, resume, fsync):
    path = tmp_path / "audit.jsonl"
    AuditLog(str(path)).append(_verdict(), action={"label": "initial"})
    _strip_final_newline(path)
    assert verify_file(str(path)) == (True, [])  # content intact: only the terminator is gone

    log = AuditLog(str(path), resume=resume, verify_on_open=True, fsync=fsync)
    entry = log.append(_verdict(), action={"label": "second"})

    entries = load(str(path))
    assert [e["action"]["label"] for e in entries] == ["initial", "second"]
    assert entry["seq"] == 1 and _entries_chained(entries)
    assert verify_file(str(path)) == (True, [])
    assert path.read_bytes().endswith(b"\n")


def test_an_already_open_log_terminates_a_tail_that_lost_its_newline_after_open(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(str(path), verify_on_open=True)
    log.append(_verdict(), action={"label": "initial"})
    _strip_final_newline(path)  # something else rewrote the file under the open object
    log.append(_verdict(), action={"label": "second"})
    assert [e["action"]["label"] for e in load(str(path))] == ["initial", "second"]
    assert verify_file(str(path)) == (True, [])


def test_the_hook_records_a_readable_decision_after_newline_loss(tmp_path):
    path = tmp_path / "audit.jsonl"
    AuditLog(str(path)).append(_verdict(), action={"label": "initial"})
    _strip_final_newline(path)
    log = AuditLog(str(path), resume="tail", verify_on_open=True, fsync=True)

    decision, _ = decide(
        "run_sql", {"sql": "SELECT 1"}, lambda *_: [], audit=log, surface="t", tool_use_id="after"
    )

    assert decision == "defer"
    entries = load(str(path))
    assert len(entries) == 2
    assert entries[1]["action"]["tool_use_id"] == "after"
    assert verify_file(str(path)) == (True, [])


# --- an incomplete record: refuse, preserve, recover ------------------------------------------


@pytest.mark.parametrize("resume", ["full", "tail"])
def test_an_incomplete_tail_refuses_the_append_and_preserves_the_bytes(tmp_path, resume):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(str(path), resume=resume, verify_on_open=True, fsync=True)
    log.append(_verdict(), action={"label": "initial"})
    good = path.read_bytes()
    with path.open("ab") as fh:  # a cooperating writer died mid-write after we opened
        fh.write(PARTIAL)

    with pytest.raises(AuditFramingError) as info:
        log.append(_verdict(), action={"label": "never"})

    assert path.read_bytes() == good + PARTIAL  # nothing written, nothing truncated
    assert info.value.offset == len(good) and info.value.length == len(PARTIAL)
    assert isinstance(info.value, AuditIntegrityError)  # existing handlers still catch it
    assert "truncate" in str(info.value) and "recusal audit verify" in str(info.value)
    assert log.last_hash == json.loads(good)["hash"]  # the in-memory head did not advance


def test_a_fresh_log_on_an_incomplete_tail_refuses_the_append(tmp_path):
    path = tmp_path / "audit.jsonl"
    AuditLog(str(path)).append(_verdict(), action={"label": "initial"})
    with path.open("ab") as fh:
        fh.write(PARTIAL)
    log = AuditLog(str(path), resume="tail")  # resume is tolerant (head recovery)...
    with pytest.raises(AuditFramingError):  # ...the append is not
        log.append(_verdict())


def test_the_hook_denies_the_tool_when_the_record_cannot_be_appended(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(str(path), resume="tail", fsync=True)
    log.append(_verdict(), action={"label": "initial"})
    with path.open("ab") as fh:
        fh.write(PARTIAL)
    ran = []

    def policy(name, args):
        ran.append(name)
        return []

    decision, reason = decide(
        "run_sql", {"sql": "SELECT 1"}, policy, audit=log, surface="t", tool_use_id="after"
    )

    assert decision == "deny"
    assert "audit log unavailable" in reason and "incomplete record" in reason
    assert ran == ["run_sql"]  # judged, then refused at the record: the record is the control
    assert len(load(str(path))) == 1 and path.read_bytes().endswith(PARTIAL)


def test_recovery_truncate_then_verify_then_resume(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(str(path))
    log.append(_verdict(), action={"label": "initial"})
    with path.open("ab") as fh:
        fh.write(PARTIAL)
    with pytest.raises(AuditFramingError) as info:
        log.append(_verdict())

    with path.open("r+b") as fh:  # the operator's documented recovery
        fh.truncate(info.value.offset)
    assert verify_file(str(path)) == (True, [])
    log.append(_verdict(), action={"label": "resumed"})
    entries = load(str(path))
    assert [e["action"]["label"] for e in entries] == ["initial", "resumed"]
    assert verify(entries)[0] and verify_file(str(path)) == (True, [])


# --- what is unchanged ------------------------------------------------------------------------


def test_a_terminated_corrupt_line_still_does_not_brick_the_append(tmp_path):
    # A corrupt line WITH its terminator has a line boundary: the chain continues past it
    # and the strict verifier names it, exactly as before.
    path = tmp_path / "audit.jsonl"
    log = AuditLog(str(path))
    log.append(_verdict(), action={"label": "initial"})
    with path.open("ab") as fh:
        fh.write(PARTIAL + b"\n")
    log.append(_verdict(), action={"label": "after"})
    assert [e["action"]["label"] for e in load(str(path))] == ["initial", "after"]
    ok, problems = verify_file(str(path))
    assert not ok and any("not valid JSON" in p for p in problems)


def test_an_empty_file_and_a_terminated_file_append_without_a_leading_newline(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_bytes(b"")
    log = AuditLog(str(path))
    log.append(_verdict(), action={"label": "a"})
    log.append(_verdict(), action={"label": "b"})
    raw = path.read_bytes()
    assert not raw.startswith(b"\n") and b"\n\n" not in raw.replace(b"\r\n", b"\n")
    assert len(load(str(path))) == 2
