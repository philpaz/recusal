"""Audit hardening: parallel writers, end-seek recovery, transactional append, strict shape.

Claude Code runs hooks for parallel tool calls concurrently, so the append path must be
one serialized transaction (lock, re-read head from the file, write, commit): several
REAL processes hammering one file must yield one continuous, verifiable chain with unique
sequential seqs - no forks, no lost records, no interleaved JSON.
"""

import errno
import subprocess
import sys
import threading

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


# --- Windows: a race on a fresh lock file's first byte must wait, not raise ----------------


class _FakeWindowsLock:
    """Stands in for ``msvcrt``: a real mutex, so the loser of the race below can be made
    to actually wait for the winner instead of the test depending on thread-scheduling
    luck."""

    LK_NBLCK = 1
    LK_UNLCK = 0

    def __init__(self):
        self._lock = threading.Lock()

    def locking(self, fd, mode, nbytes):
        if mode == self.LK_UNLCK:
            self._lock.release()
            return
        if not self._lock.acquire(blocking=False):
            raise OSError(errno.EACCES, "locked")


class _FirstByteRace:
    """Exactly one writer "wins" the one-time write of a fresh lock file's first byte;
    every other writer gets the ``PermissionError`` a real Windows writer gets when its
    write lands on a byte another handle already holds locked (issue #67)."""

    def __init__(self):
        self._claim_lock = threading.Lock()
        self._claimed = False

    def claim_or_raise(self):
        with self._claim_lock:
            if self._claimed:
                raise PermissionError(
                    errno.EACCES,
                    "The process cannot access the file because another process has "
                    "locked a portion of the file",
                )
            self._claimed = True


class _RacingLockFile:
    """Wraps a real file handle so two threads' first ``tell()`` (the "is this file
    still empty" check in ``_interprocess_lock``) line up before either writes, forcing
    the race every run instead of leaving it to thread-scheduling luck."""

    def __init__(self, fh, race, barrier):
        self._fh = fh
        self._race = race
        self._barrier = barrier
        self._told = False

    def __getattr__(self, name):
        return getattr(self._fh, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return self._fh.__exit__(*exc_info)

    def tell(self):
        if not self._told:
            self._told = True
            self._barrier.wait()  # both writers must see "empty" before either writes
        return self._fh.tell()

    def write(self, data):
        if data == b"\0":
            self._race.claim_or_raise()
        return self._fh.write(data)


def test_concurrent_first_appends_to_a_fresh_lock_file_do_not_raise(tmp_path, monkeypatch):
    """Issue #67: on Windows, the first append from several writers can land at the same
    instant; all of them see a fresh, empty ``<path>.lock`` and all try to write its
    one-time byte. Only one write can land - today the losers' write raises
    PermissionError straight out of ``_interprocess_lock`` instead of being treated as
    "someone else already created the byte", the same contention ``_acquire_windows_lock``
    already waits out for the lock itself.
    """
    from recusal import audit
    from recusal.audit import _interprocess_lock

    monkeypatch.setattr(audit.sys, "platform", "win32")
    monkeypatch.setattr(audit, "msvcrt", _FakeWindowsLock(), raising=False)

    for round_number in range(5):
        lock_path = str(tmp_path / f"audit{round_number}.jsonl.lock")
        race = _FirstByteRace()
        barrier = threading.Barrier(2)
        real_open = open

        def fake_open(path, mode="r", *args, **kwargs):
            fh = real_open(path, mode, *args, **kwargs)
            if path == lock_path and mode == "a+b":
                return _RacingLockFile(fh, race, barrier)
            return fh

        monkeypatch.setattr(audit, "open", fake_open, raising=False)

        errors = []

        def run():
            try:
                with _interprocess_lock(lock_path):
                    pass
            except Exception as exc:  # noqa: BLE001 - captured across a thread boundary
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
            assert not t.is_alive()

        assert not errors, (
            f"round {round_number}: a concurrent first append raised instead of waiting: {errors!r}"
        )


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


@pytest.mark.parametrize("record", ["[]", "null", '"string"', "42"])
def test_expected_head_with_a_non_object_final_record_is_a_failure_not_a_crash(tmp_path, record):
    # P1-2 regression: the anchor exists precisely to catch a mangled tail, so a
    # non-object final record must be a NAMED head mismatch, never an AttributeError.
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path=str(path))
    entry = log.append(_verdict("real"))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(record + "\n")
    intact, problems = verify_file(str(path), expected_head=(2, entry["hash"]))
    assert not intact
    assert any("head mismatch" in p or "not an audit entry" in p for p in problems)
    intact, problems = verify(load(str(path)), expected_head=(2, entry["hash"]))
    assert not intact


def test_expected_head_on_an_entirely_non_object_log():
    intact, problems = verify([[]], expected_head=(1, "0" * 64))
    assert not intact and problems
