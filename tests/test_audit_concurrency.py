"""Audit hardening: parallel writers, end-seek recovery, transactional append, strict shape.

Claude Code runs hooks for parallel tool calls concurrently, so the append path must be
one serialized transaction (lock, re-read head from the file, write, commit): several
REAL processes hammering one file must yield one continuous, verifiable chain with unique
sequential seqs - no forks, no lost records, no interleaved JSON.
"""

import subprocess
import sys

import pytest

from recusal import AuditLog, compute_verdict, verify
from recusal.audit import _tail_state, load, verify_file


def _verdict(msg="ok"):
    return compute_verdict([{"check": "c", "severity": "INFO", "status": "pass", "message": msg}])


# --- parallel writers: one chain, not siblings ---------------------------------------------

_WRITER = """
import sys
from recusal import AuditLog, compute_verdict

path, writer_id, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
for i in range(count):
    v = compute_verdict([{"check": "c", "severity": "INFO", "status": "pass"}])
    AuditLog(path=path, resume="tail").append(v, action={"writer": writer_id, "i": i})
"""


def test_parallel_processes_extend_one_chain(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    writers, per_writer = 4, 8
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WRITER, path, str(w), str(per_writer)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for w in range(writers)
    ]
    for p in procs:
        _, err = p.communicate(timeout=120)
        assert p.returncode == 0, err.decode(errors="replace")

    entries = load(path)
    assert len(entries) == writers * per_writer  # no lost records
    assert [e["seq"] for e in entries] == list(range(writers * per_writer))  # unique, gapless
    ok, problems = verify(entries)
    assert ok, problems  # one continuous chain, no siblings
    ok, problems = verify_file(path)
    assert ok, problems  # and no interleaved/corrupt JSON lines


# --- end-seek head recovery ----------------------------------------------------------------


def test_tail_state_recovers_the_head_from_the_final_record(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    log = AuditLog(path=path)
    for i in range(20):
        log.append(_verdict(str(i)))
    assert _tail_state(path) == (20, log.last_hash)


def test_tail_state_walks_past_a_corrupt_trailing_line(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path=str(path))
    log.append(_verdict("intact"))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"seq": 1, "truncat')  # killed writer: partial, unparseable
    assert _tail_state(str(path)) == (1, log.last_hash)


def test_tail_state_falls_back_on_a_parseable_non_entry_tail(tmp_path):
    # A parseable line that is NOT a usable entry must not be walked past (that would
    # recover a stale head); accounting falls back to the full-scan semantics.
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path=str(path))
    log.append(_verdict("intact"))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("[]\n")
    next_seq, last_hash = _tail_state(str(path))
    assert next_seq == 2  # the non-entry line counts, exactly as resume="full" counts it
    assert last_hash == log.last_hash


def test_tail_state_of_missing_and_empty_files(tmp_path):
    from recusal.audit import GENESIS

    assert _tail_state(str(tmp_path / "nope.jsonl")) == (0, GENESIS)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert _tail_state(str(empty)) == (0, GENESIS)


# --- transactional append: a failed write never advances the chain -------------------------


def test_a_failed_write_does_not_advance_in_memory_state(tmp_path, monkeypatch):
    path = str(tmp_path / "audit.jsonl")
    log = AuditLog(path=path)
    log.append(_verdict("one"))
    seq_before, head_before = log._next_seq, log.last_hash

    import builtins

    real_open = builtins.open

    def _failing_open(file, *args, **kwargs):
        if str(file) == path and args and "a" in args[0]:
            raise OSError("disk full")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _failing_open)
    with pytest.raises(OSError):
        log.append(_verdict("two"))
    monkeypatch.undo()

    assert (log._next_seq, log.last_hash) == (seq_before, head_before)
    entry = log.append(_verdict("three"))  # the sink recovered; the chain continues
    assert entry["seq"] == seq_before
    ok, problems = verify_file(path)
    assert ok, problems


def test_fsync_append_roundtrips(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    log = AuditLog(path=path, fsync=True)
    log.append(_verdict("durable"))
    ok, problems = verify_file(path)
    assert ok, problems


# --- strict verification of valid-JSON non-entries ------------------------------------------


@pytest.mark.parametrize("line", ["[]", "null", '"string"', "42"])
def test_verify_file_names_valid_json_non_entries(tmp_path, line):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path=str(path))
    log.append(_verdict("real"))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    intact, problems = verify_file(str(path))
    assert not intact
    assert any("not an audit entry object" in p for p in problems)


def test_verify_handles_non_dict_entries_without_raising():
    intact, problems = verify([[], None, "x", 42])
    assert not intact and len(problems) >= 4


def test_verify_file_reports_a_directory_as_unreadable(tmp_path):
    intact, problems = verify_file(str(tmp_path))
    assert not intact
    assert any("cannot read" in p for p in problems)


def test_verify_file_reports_invalid_utf8_as_unreadable(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_bytes(b'{"seq": 0}\n\xff\xfe\n')
    intact, problems = verify_file(str(path))
    assert not intact
    assert any("cannot read" in p for p in problems)


def test_verify_flags_malformed_hash_shapes():
    entry = {"seq": 0, "prev_hash": "0" * 64, "hash": "not-a-digest", "decision": "PASS"}
    intact, problems = verify([entry])
    assert not intact
    assert any("not a sha256 hex digest" in p for p in problems)
