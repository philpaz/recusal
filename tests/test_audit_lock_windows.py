"""The Windows inter-process audit lock: contention is waited out, not turned into a
ten-second cliff, and the deadline still fails closed.

``msvcrt.locking(LK_LOCK)`` retries ten times one second apart and then raises
``PermissionError``; on a loaded runner four concurrent hook processes exhausted that on
2026-08-26 (CI run 33018557987, windows-latest). The lock now polls ``LK_NBLCK`` with a
short sleep until a generous deadline, retrying ONLY on the contention errno. These tests
drive that loop directly through a fake ``msvcrt`` so they are deterministic and run on
every platform; one exercises the real lock where it exists.
"""

import errno
import sys

import pytest

from recusal import audit


class _FakeMsvcrt:
    """Refuses the first ``refusals`` attempts with the contention errno, then grants."""

    LK_NBLCK = 1
    LK_UNLCK = 0
    LK_LOCK = 2

    def __init__(self, refusals, error=errno.EACCES):
        self.refusals = refusals
        self.error = error
        self.attempts = 0
        self.unlocked = 0

    def locking(self, fd, mode, nbytes):
        if mode == self.LK_UNLCK:
            self.unlocked += 1
            return
        self.attempts += 1
        if self.attempts <= self.refusals:
            raise OSError(self.error, "locked")


@pytest.fixture
def lock_file(tmp_path):
    with open(tmp_path / "audit.jsonl.lock", "a+b") as fh:
        fh.write(b"\0")
        fh.flush()
        yield fh


def test_contention_is_waited_out_rather_than_failing_at_a_cliff(monkeypatch, lock_file):
    fake = _FakeMsvcrt(refusals=25)  # more than LK_LOCK's ten one-second retries
    monkeypatch.setattr(audit, "msvcrt", fake, raising=False)
    monkeypatch.setattr(audit, "_LOCK_RETRY_SECONDS", 0.0)
    audit._acquire_windows_lock(lock_file, "audit.jsonl.lock")
    assert fake.attempts == 26


def test_the_deadline_still_fails_closed(monkeypatch, lock_file):
    fake = _FakeMsvcrt(refusals=10**9)
    monkeypatch.setattr(audit, "msvcrt", fake, raising=False)
    monkeypatch.setattr(audit, "_LOCK_RETRY_SECONDS", 0.0)
    monkeypatch.setattr(audit, "_LOCK_TIMEOUT_SECONDS", 0.02)
    with pytest.raises(OSError, match="refusing to append off the record"):
        audit._acquire_windows_lock(lock_file, "audit.jsonl.lock")
    assert fake.attempts >= 1


def test_only_contention_is_retried(monkeypatch, lock_file):
    """A bad descriptor or an unsupported handle is not contention: it must surface at
    once, never spin to the deadline (which is how a BytesIO nearly hid this)."""
    fake = _FakeMsvcrt(refusals=10**9, error=errno.EBADF)
    monkeypatch.setattr(audit, "msvcrt", fake, raising=False)
    monkeypatch.setattr(audit, "_LOCK_RETRY_SECONDS", 0.0)
    with pytest.raises(OSError):
        audit._acquire_windows_lock(lock_file, "audit.jsonl.lock")
    assert fake.attempts == 1


@pytest.mark.skipif(sys.platform != "win32", reason="the real msvcrt lock only exists on Windows")
def test_real_lock_round_trips_on_windows(tmp_path):
    from recusal.audit import _interprocess_lock

    lock = str(tmp_path / "audit.jsonl.lock")
    with _interprocess_lock(lock):
        pass
    with _interprocess_lock(lock):  # re-acquirable after release
        pass


@pytest.mark.skipif(sys.platform != "win32", reason="exercises the Windows branch")
def test_a_locked_out_append_is_a_failed_append_not_a_silent_skip(tmp_path, monkeypatch):
    """The hook wiring turns a raised append into a deny; the lock must raise, never
    return without the lock."""
    from recusal import compute_verdict

    fake = _FakeMsvcrt(refusals=10**9)
    monkeypatch.setattr(audit, "msvcrt", fake)
    monkeypatch.setattr(audit, "_LOCK_RETRY_SECONDS", 0.0)
    monkeypatch.setattr(audit, "_LOCK_TIMEOUT_SECONDS", 0.02)
    log = audit.AuditLog(path=str(tmp_path / "audit.jsonl"))
    with pytest.raises(OSError):
        log.append(compute_verdict([]))
    assert log.entries == [] and log._next_seq == 0  # nothing advanced off the record
