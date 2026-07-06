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
- It is **single-writer**: two processes appending to the same file will fork the chain.
- Resuming an existing file does **not** re-verify it; run ``verify_file`` first if you
  need to know the log you are extending is intact.

Deterministic and dependency-free: SHA-256 over canonical JSON, standard library only.
The record shape maps cleanly onto OWASP Agentic logging and EU AI Act Article 12
(record-keeping).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from .evidence import Verdict

GENESIS = "0" * 64  # the prev_hash of the first entry


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
    """

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.path = path
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.entries: List[Dict[str, Any]] = []
        self.last_hash = GENESIS
        if path:
            for entry in load(path):
                self.entries.append(entry)
                head = entry.get("hash")
                if head is not None:
                    self.last_hash = head

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
        optional agent/session id. Returns the written entry (including its hash)."""
        entry: Dict[str, Any] = {
            "seq": len(self.entries),
            "timestamp": timestamp or self._clock().isoformat(),
            "actor": actor,
            "action": action,
            **_summarize(verdict),
            "prev_hash": self.last_hash,
        }
        entry["hash"] = _digest(entry)

        self.entries.append(entry)
        self.last_hash = entry["hash"]
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=True, default=str) + "\n")
        return entry


def load(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL audit file into a list of entries (empty if the file is absent).

    A line that is not valid JSON (e.g. a partial final line left by a process killed
    mid-append) is skipped rather than crashing the read, so a half-written tail can never
    brick the log. ``verify`` still surfaces any resulting chain break.
    """
    entries: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # skip a corrupt/partial line instead of bricking the log
    except FileNotFoundError:
        return []
    return entries


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
        seq = entry.get("seq")
        if seq != i:
            problems.append(f"entry {i} (seq {seq}): out of order or gap - seq != position")
        if entry.get("prev_hash") != prev:
            problems.append(f"entry {i} (seq {seq}): broken link - prev_hash does not match")
        if entry.get("hash") != _digest(entry):
            problems.append(f"entry {i} (seq {seq}): content tampered - hash does not match")
        prev = entry.get("hash") or ""
    if expected_head is not None:
        count, last_hash = expected_head
        if len(entries) != count:
            problems.append(
                f"length mismatch: {len(entries)} entries, anchor expects {count} (possible truncation)"
            )
        actual_head = entries[-1].get("hash") if entries else GENESIS
        if actual_head != last_hash:
            problems.append("head mismatch: last hash does not match the external anchor")
    return (not problems), problems


def verify_file(path: str) -> Tuple[bool, List[str]]:
    """Convenience: load a JSONL audit file and verify its chain."""
    return verify(load(path))
