"""
Hash-chained audit log, a linked record of every verdict.

A gate that can refuse is only half of an auditable control; the other half is a
record you can replay and an auditor can read. ``recusal.audit`` appends each verdict to an
append-only, hash-chained log: every entry carries the SHA-256 hash of the entry
before it, so an **in-place edit or a reordering** of any entry that has at least one
untampered successor breaks the chain, and ``verify`` catches it, naming the entry and
the reason.

What this does and does not guarantee (read before relying on it):

- It detects in-place edits and reordering **of any entry that still has an untampered
  entry after it**. It is tamper-**evident**, not tamper-proof.
- The digest is **unkeyed** and the head is **unanchored**, so an attacker with write
  access to the file can rewrite any **tail suffix** (in the limit, just the last entry,
  the most recently recorded action), recomputing only that suffix's hashes, or truncate
  the tail, and still pass ``verify``. Anyone with write access can also append a valid new
  entry; the log proves chain consistency, not who wrote it. To catch all of these, commit
  the head ``(count, last_hash)`` somewhere the attacker cannot also rewrite (a witness, a
  WORM store, a signature) and pass it as ``verify(..., expected_head=...)``.
- File-backed appends are **serialized with an inter-process lock** (``<path>.lock``)
  and re-derive the chain head from the file under that lock, so concurrent writers
  (e.g. hooks for parallel tool calls) extend one chain instead of forking it. The
  in-memory ``entries`` mirror is still per-process: under concurrency, verify the
  FILE (``verify_file``), not one process's mirror.
- Resuming an existing file does **not** re-verify it by default. Pass
  ``verify_on_open=True`` to refuse (``AuditIntegrityError``) to extend a file that fails
  strict verification, and ``expected_head=(count, last_hash)`` to also anchor the resume
  against truncation, tail-suffix rewrite, and forged appends.
- To keep such an anchor, or to mirror the record into a store the attacker cannot
  rewrite (WORM, another host), pass ``sinks=[...]``: every committed entry is handed to
  each :class:`AuditSink`, and a sink failure surfaces as a failed append (fail closed).
- **The log contains what you put in it.** ``tool_input`` is never embedded by the hook
  wiring (fingerprint only), but finding *messages*, verdict reasons, and exception text
  are stored in plaintext - a policy that writes a secret or a regulated value into a
  message has put it on the record. Keep payloads out of messages.

Deterministic and dependency-free: SHA-256 over canonical JSON, standard library only.
The record shape maps cleanly onto OWASP Agentic logging and EU AI Act Article 12
(record-keeping).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple

from .evidence import Verdict

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

GENESIS = "0" * 64  # the prev_hash of the first entry

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class AuditIntegrityError(Exception):
    """Raised when ``verify_on_open`` (or an ``expected_head`` anchor) refuses a resume.

    Extending a log that already fails verification would hash-chain new entries onto a
    tampered or truncated record, laundering the break behind fresh valid links. So the
    refusal happens at construction, before any resume state exists: the caller gets no
    log object to append to. ``problems`` carries the verifier's findings verbatim.
    """

    def __init__(self, path: str, problems: List[str]) -> None:
        super().__init__(f"audit log {path!r} failed verification on open: " + "; ".join(problems))
        self.path = path
        self.problems = problems


class AuditSink(Protocol):
    """Structural protocol for mirroring committed audit entries somewhere else.

    The chain in the local file is tamper-evident, not tamper-proof: an attacker with
    write access can rewrite a tail suffix or truncate it (see the module docstring). A
    sink is the extension point for the countermeasure - hand each committed entry to a
    store that attacker cannot also rewrite (a WORM bucket, another host, a signer).
    Any object with a matching ``write`` method qualifies; no registration, no base class.

    Contract:

    - ``write(entry)`` receives the entry dict AFTER the local record is committed,
      in chain order (on a file-backed log it is called inside the append lock, so a
      shared sink sees one global order even under concurrent writers).
    - Each entry carries its own ``seq`` and ``hash``, so a sink can maintain the
      external head anchor directly: ``(entry["seq"] + 1, entry["hash"])`` is exactly
      the ``expected_head`` that ``verify``/``verify_file``/``verify_on_open`` accept.
    - A sink signals failure by raising. The exception propagates out of ``append`` so
      the caller fails closed (the hook wiring already turns that into a deny); the
      local entry remains on record, and a retry appends a NEW entry with a new seq -
      a sink must tolerate a seq gap, never assume it saw every attempt.
    """

    def write(self, entry: Dict[str, Any]) -> None:
        """Record one committed entry; raise to report failure."""


@contextmanager
def _interprocess_lock(lock_path: str) -> Iterator[None]:
    """Exclusive cross-process lock on ``lock_path`` (created on first use).

    Claude Code runs hooks for parallel tool calls concurrently, so two short-lived hook
    processes can append to the same audit file at the same moment; without a lock both
    read the same head and write sibling entries, forking the chain with neither append
    reporting an error. The lock serializes the whole read-head-plus-append transaction.
    On Windows an uncontended-timeout failure raises, which a caller must treat as a
    failed append (fail closed), never a silent skip.
    """
    with open(lock_path, "a+b") as fh:
        if sys.platform == "win32":
            fh.seek(0, os.SEEK_END)
            if fh.tell() == 0:
                fh.write(b"\0")
                fh.flush()
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _tail_state(path: str) -> Tuple[int, str]:
    """Recover ``(next_seq, last_hash)`` from the END of a JSONL audit file.

    Seeks backward in growing windows to the last nonblank line that parses as an entry
    object carrying a usable ``seq`` and ``hash``, so the cost is proportional to the
    final record, not the log (a corrupt trailing line from a killed writer is walked
    past, the same tolerance as ``_scan``). A log whose trailing entries never yield a
    usable head falls back to one forward scan.
    """

    def _full_scan() -> Tuple[int, str]:
        # Matches the resume="full" accounting exactly: every parseable line counts
        # toward seq; the head is the last entry that carries a hash.
        next_seq, last_hash = 0, GENESIS
        for entry in _scan(path):
            next_seq += 1
            head = entry.get("hash") if isinstance(entry, dict) else None
            if isinstance(head, str) and head:
                last_hash = head
        return next_seq, last_hash

    try:
        size = os.path.getsize(path)
    except OSError:
        return 0, GENESIS
    window = 64 * 1024
    with open(path, "rb") as fh:
        while size > 0:
            start = max(0, size - window)
            fh.seek(start)
            lines = fh.read(size - start).split(b"\n")
            if start > 0:
                lines = lines[1:]  # the first piece may be a partial line; drop it
            for raw in reversed(lines):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    continue  # corrupt/partial tail line: keep walking backward
                if (
                    isinstance(entry, dict)
                    and isinstance(entry.get("seq"), int)
                    and entry["seq"] >= 0
                    and isinstance(entry.get("hash"), str)
                    and entry["hash"]
                ):
                    return entry["seq"] + 1, entry["hash"]
                # Parseable but not a usable entry: walking PAST it could recover a
                # stale head and fork the chain; only a full scan accounts for it.
                return _full_scan()
            if start == 0:
                break
            window *= 4
    return _full_scan()


def _canonical(entry: Dict[str, Any]) -> str:
    """Stable serialization of an entry for hashing, everything but the hash field."""
    payload = {k: v for k, v in entry.items() if k != "hash"}
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )


def _digest(entry: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(entry).encode("utf-8")).hexdigest()


def _summarize(verdict: Verdict) -> Dict[str, Any]:
    return {
        "decision": verdict.decision.value,
        "highest_severity": verdict.highest_severity.value,
        "reasons": verdict.reasons(),
        "failures": [
            {"check": f.check, "severity": f.severity.value, "message": f.message}
            for f in verdict.failures
        ],
    }


class AuditLog:
    """Append-only, hash-chained log of verdicts.

    Pass ``path`` to persist as JSONL (one entry per line); otherwise it lives in
    memory on ``self.entries``. An existing file is resumed, the chain continues
    from its last entry.

    ``resume`` controls what a resume holds in memory:

    - ``"full"`` (default): load every prior entry into ``self.entries`` and keep
      every new append there too, so ``verify(log.entries)`` works directly. The
      cost is the whole log in memory - fine for a short-lived process (a per-call
      hook), unbounded for a long-running gate over a growing log.
    - ``"tail"``: recover the chain head (last hash, next seq) from the END of the
      file (backward seek to the final usable record, full-scan fallback only for
      pathological logs) and retain **no** entries in memory, before or after -
      appends go to disk only and ``self.entries`` stays empty. Bounded memory and
      final-record-proportional time regardless of log size; verify with
      ``verify_file(path)``. Requires ``path``.

    ``verify_on_open=True`` runs the strict verifier over an existing file BEFORE
    resuming and raises :class:`AuditIntegrityError` if it is not intact - you never
    extend a log you cannot trust. An absent file is a new log, not a failure (absence
    is only tamper-evidence against an anchor). ``expected_head=(count, last_hash)``
    supplies that anchor: it implies ``verify_on_open`` and additionally refuses
    truncation, tail-suffix rewrite, and forged appends, including a missing file
    unless the anchor itself says empty (``(0, GENESIS)``). Verification reads the
    whole file once, even with ``resume="tail"`` (integrity is a whole-chain property).
    Both require ``path``.

    ``sinks`` (a list or tuple of :class:`AuditSink`) mirrors every committed entry to
    external stores; see the protocol's contract for ordering and failure semantics.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        clock: Optional[Callable[[], datetime]] = None,
        resume: str = "full",
        fsync: bool = False,
        sinks: Optional[Sequence[AuditSink]] = None,
        verify_on_open: bool = False,
        expected_head: Optional[Tuple[int, str]] = None,
    ) -> None:
        if resume not in ("full", "tail"):
            raise ValueError(f"resume must be 'full' or 'tail', got {resume!r}")
        if resume == "tail" and not path:
            raise ValueError("resume='tail' requires a path (an in-memory log IS its entries)")
        if sinks is None:
            self._sinks: Tuple[AuditSink, ...] = ()
        else:
            # Strict container discipline (same doctrine as the manifest verifier): a
            # sink that silently is not one would drop the mirror without an error.
            if not isinstance(sinks, (list, tuple)):
                raise ValueError(
                    f"sinks must be a list or tuple of AuditSink objects, got {type(sinks).__name__}"
                )
            for sink in sinks:
                if not callable(getattr(sink, "write", None)):
                    raise ValueError(
                        f"sink {sink!r} has no callable write(entry) method (see AuditSink)"
                    )
            self._sinks = tuple(sinks)
        if expected_head is not None:
            verify_on_open = True  # an anchor you do not check is not an anchor
        if verify_on_open and not path:
            raise ValueError("verify_on_open requires a path (an in-memory log starts empty)")
        if path and verify_on_open:
            if not os.path.exists(path):
                if expected_head is not None:
                    count, last_hash = expected_head
                    if count != 0 or last_hash != GENESIS:
                        raise AuditIntegrityError(
                            path,
                            [
                                f"no audit log at {path!r} but the anchor expects "
                                f"{count} entries - a missing log is not an intact log"
                            ],
                        )
                # no file and no (non-empty) anchor: a NEW log, nothing to verify yet
            else:
                intact, problems = verify_file(path, expected_head=expected_head)
                if not intact:
                    raise AuditIntegrityError(path, problems)
        self.path = path
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._retain = resume == "full"
        self._fsync = fsync
        self.entries: List[Dict[str, Any]] = []
        self.last_hash = GENESIS
        self._next_seq = 0
        if path:
            if self._retain:
                for entry in _scan(path):
                    self.entries.append(entry)
                    self._next_seq += 1
                    head = entry.get("hash") if isinstance(entry, dict) else None
                    if head is not None:
                        self.last_hash = head
            else:
                self._next_seq, self.last_hash = _tail_state(path)

    def _build(
        self, verdict: Verdict, seq: int, prev_hash: str, action: Any, actor: Optional[str], ts: str
    ) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "seq": seq,
            "timestamp": ts,
            "actor": actor,
            "action": action,
            **_summarize(verdict),
            "prev_hash": prev_hash,
        }
        entry["hash"] = _digest(entry)
        return entry

    def append(
        self,
        verdict: Verdict,
        *,
        action: Any = None,
        actor: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record one verdict. ``action`` is any JSON-serializable description of what
        was adjudicated (e.g. ``{"tool": "Bash", "command": "..."}``); ``actor`` is an
        optional agent/session id. Returns the written entry (including its hash).

        A file-backed append is one serialized transaction: an inter-process lock
        (``<path>.lock``) is held while the chain head is re-read from the END of the
        file and the new entry is written, so concurrent writers (Claude Code runs hooks
        for parallel tool calls concurrently) extend one chain instead of forking it.
        In-memory state commits only AFTER the write succeeds: a failed write raises and
        never advances the chain. ``fsync=True`` additionally forces the entry to stable
        storage before the lock is released.

        Sinks are notified AFTER the local record is committed (on a file-backed log,
        inside the same lock, so a shared sink sees entries in chain order even under
        concurrent writers). A sink that raises makes this call raise - the caller must
        fail closed - while the local entry remains on record; the in-memory head/mirror
        is then stale until the next append re-derives it from the file, which is why
        under concurrency you verify the FILE, not one process's mirror.
        """
        ts = timestamp or self._clock().isoformat()
        if self.path:
            with _interprocess_lock(self.path + ".lock"):
                # The file is the shared truth under concurrency; our in-memory head may
                # be behind another process's appends, so it is re-derived here.
                next_seq, last_hash = _tail_state(self.path)
                entry = self._build(verdict, next_seq, last_hash, action, actor, ts)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=True, default=str) + "\n")
                    fh.flush()
                    if self._fsync:
                        os.fsync(fh.fileno())
                for sink in self._sinks:
                    sink.write(entry)
            self._next_seq = next_seq + 1
            self.last_hash = entry["hash"]
            if self._retain:
                self.entries.append(entry)
            return entry
        entry = self._build(verdict, self._next_seq, self.last_hash, action, actor, ts)
        self._next_seq += 1
        self.entries.append(entry)
        self.last_hash = entry["hash"]
        for sink in self._sinks:
            sink.write(entry)
        return entry


def _scan(path: str) -> Iterator[Dict[str, Any]]:
    """Yield entries from a JSONL audit file one at a time (nothing if it is absent).

    A line that is not valid JSON (e.g. a partial final line left by a process killed
    mid-append) is skipped rather than crashing the read, so a half-written tail can never
    brick the log. ``verify`` still surfaces any resulting chain break.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip a corrupt/partial line instead of bricking the log
    except FileNotFoundError:
        return


def load(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL audit file into a list of entries (empty if the file is absent).

    Corrupt/partial lines are skipped, not fatal - see ``_scan``.
    """
    return list(_scan(path))


def verify(
    entries: List[Dict[str, Any]],
    *,
    expected_head: Optional[Tuple[int, str]] = None,
) -> Tuple[bool, List[str]]:
    """Check the hash chain. Returns ``(intact, problems)``, ``problems`` is empty when the
    log is intact, otherwise it names each broken entry and why.

    This detects in-place edits and reordering of any entry that still has an untampered
    successor. It does **not**, on its own, detect truncation of the tail, a rewrite of any
    tail suffix (in the limit, just the last entry, recomputing only that suffix's hashes),
    or a valid forged append (the digest is unkeyed and the head is unanchored). Pass
    ``expected_head=(count, last_hash)``, a value committed somewhere the attacker cannot
    also rewrite, to catch truncation, tail-suffix rewrite, and forged appends.
    """
    problems: List[str] = []
    prev: str = GENESIS
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            # valid JSON is not necessarily an audit entry; a verifier must name that,
            # not crash out of a verdict (a crash is not a verification result).
            problems.append(
                f"entry {i}: not an audit entry object ({type(entry).__name__}) - the log "
                "carries a record this verifier cannot even inspect"
            )
            prev = ""
            continue
        seq = entry.get("seq")
        if not isinstance(seq, int) or seq < 0:
            problems.append(f"entry {i} (seq {seq!r}): seq is not a nonnegative integer")
        if seq != i:
            problems.append(f"entry {i} (seq {seq}): out of order or gap - seq != position")
        if not isinstance(entry.get("hash"), str) or not _HEX64.fullmatch(entry.get("hash") or ""):
            problems.append(f"entry {i} (seq {seq}): hash is not a sha256 hex digest")
        if entry.get("prev_hash") != prev:
            problems.append(f"entry {i} (seq {seq}): broken link - prev_hash does not match")
        if entry.get("hash") != _digest(entry):
            problems.append(f"entry {i} (seq {seq}): content tampered - hash does not match")
        if entry.get("decision") not in ("PASS", "RETRY", "FAIL"):
            problems.append(
                f"entry {i} (seq {seq}): unrecognized decision {entry.get('decision')!r}"
            )
        prev = entry.get("hash") or ""
    if expected_head is not None:
        count, last_hash = expected_head
        if len(entries) != count:
            problems.append(
                f"length mismatch: {len(entries)} entries, anchor expects {count} (possible truncation)"
            )
        actual_head: Optional[str]
        if not entries:
            actual_head = GENESIS
        elif isinstance(entries[-1], dict):
            actual_head = entries[-1].get("hash")
        else:
            # a non-object final record has no head; that is a mismatch to NAME, not a
            # crash to raise (the anchor exists precisely to catch a mangled tail)
            actual_head = None
        if actual_head != last_hash:
            problems.append("head mismatch: last hash does not match the external anchor")
    return (not problems), problems


def verify_file(
    path: str, *, expected_head: Optional[Tuple[int, str]] = None
) -> Tuple[bool, List[str]]:
    """Load a JSONL audit file and verify its chain, **strictly**.

    Reading and verifying have different duties. ``load`` (and a resume) is tolerant: a
    half-written tail must not brick an application reading the earlier entries. A
    *verifier* must not share that tolerance - skipping an unreadable line and blessing
    the rest would certify a log whose most recent entries are unreadable or tampered.
    So here a nonblank line that does not parse as JSON is a verification failure, and a
    missing file is a failure too: a missing log is not an intact log.
    """
    entries: List[Dict[str, Any]] = []
    bad_lines: List[int] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    bad_lines.append(lineno)
    except FileNotFoundError:
        return False, [f"no audit log at {path!r} - a missing log is not an intact log"]
    except (OSError, UnicodeDecodeError) as exc:
        # permission denied, a directory, invalid UTF-8, an I/O failure mid-read: an
        # operational inability to inspect the log must not read as an intact log.
        return False, [f"cannot read audit log at {path!r}: {exc}"]
    intact, problems = verify(entries, expected_head=expected_head)
    if bad_lines:
        intact = False
        shown = ", ".join(str(n) for n in bad_lines[:5])
        more = ", ..." if len(bad_lines) > 5 else ""
        problems.append(
            f"{len(bad_lines)} nonblank line(s) are not valid JSON (line {shown}{more}) - "
            "the log's most recent entries may be unreadable or tampered"
        )
    return intact, problems
