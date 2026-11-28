"""Recipe 24: audit-log refusals and operator recovery, entirely offline.

Run: python examples/audit_operations.py

This demonstration damages only logs it creates in a temporary directory. Stop
all writers before investigating or recovering a real log, and preserve the
damaged file first. A framing error names an incomplete tail that can be removed;
an integrity failure requires investigation and a trusted recovery source. Never
reset an external head to make a truncated local log pass verification.

The trusted copies and heads below live in memory only to keep the example small.
Production copies and anchors must be durable and outside the agent's write reach.
These functions are a teaching sequence, not a concurrent repair utility.
"""

import json
import os
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recusal import AuditFramingError, AuditIntegrityError, AuditLog, compute_verdict
from recusal.audit import verify_file

PARTIAL = b'{"seq": 1, "interrupted":'


def _seed(path, count):
    # An accidental rerun must not damage or append to a pre-existing operator log.
    with path.open("xb"):
        pass
    log = AuditLog(str(path), verify_on_open=True)
    for number in range(count):
        log.append(compute_verdict([]), action=f"original-{number}")
    return log


def _verified(path, expected_head=None):
    intact, problems = verify_file(str(path), expected_head=expected_head)
    if not intact:
        raise AuditIntegrityError(str(path), problems)


def _backup(path):
    backup = path.with_suffix(".damaged")
    # Never replace earlier evidence or follow an existing backup symlink.
    with backup.open("xb") as destination, path.open("rb") as source:
        shutil.copyfileobj(source, destination)
    return backup


def framing_example(directory):
    """Keep the damaged bytes, remove only the reported tail, verify, then resume."""
    path = directory / "framing.jsonl"
    log = _seed(path, 1)
    with path.open("ab") as stream:
        stream.write(PARTIAL)
    damaged = path.read_bytes()

    try:
        log.append(compute_verdict([]), action="must-not-append")
    except AuditFramingError as error:
        refusal = str(error)
        offset = error.offset
    else:
        raise AssertionError("incomplete record unexpectedly accepted")
    assert path.read_bytes() == damaged

    backup = _backup(path)  # preserve evidence BEFORE changing the local log
    with path.open("r+b") as stream:
        stream.truncate(offset)
    _verified(path)  # a framing repair does not imply that the earlier chain is intact
    resumed = AuditLog(str(path), verify_on_open=True)
    resumed.append(compute_verdict([]), action="resumed")
    _verified(path)
    print(f"FRAMING REFUSED: {refusal}")
    print(f"Recovered after backup {backup.name}, truncate to {offset}, and strict verify.")
    return path, backup, offset


def integrity_example(directory):
    """Investigate a hash mismatch; restore a trusted copy instead of editing hashes."""
    path = directory / "integrity.jsonl"
    _seed(path, 2)
    trusted = path.read_bytes()  # stand-in for a protected, independently trusted copy
    lines = trusted.splitlines()
    changed = json.loads(lines[0])
    changed["actor"] = "unexpected-actor"
    lines[0] = json.dumps(changed).encode("utf-8")
    path.write_bytes(b"\n".join(lines) + b"\n")
    damaged = path.read_bytes()

    try:
        AuditLog(str(path), verify_on_open=True)
    except AuditIntegrityError as error:
        refusal = str(error)
    else:
        raise AssertionError("changed record unexpectedly accepted")
    assert path.read_bytes() == damaged

    backup = _backup(path)
    print(f"INTEGRITY REFUSED: {refusal}")
    print("Investigate; do not delete records or recompute hashes to conceal the mismatch.")
    # For this synthetic case we know what changed and retained the original bytes.
    # An operator without a trusted recovery source must keep the gate stopped.
    path.write_bytes(trusted)
    _verified(path)
    resumed = AuditLog(str(path), verify_on_open=True)
    resumed.append(compute_verdict([]), action="resumed")
    _verified(path)
    print("Restored the trusted copy, verified it, and resumed; damaged copy retained.")
    return path, backup


def anchor_example(directory):
    """A valid prefix still fails its protected head; restore data, not the anchor."""
    path = directory / "anchor.jsonl"
    log = _seed(path, 2)
    trusted = path.read_bytes()
    expected_head = (2, log.last_hash)  # stand-in for an independently protected witness
    path.write_bytes(trusted.splitlines(keepends=True)[0])
    damaged = path.read_bytes()
    _verified(path)  # the remaining prefix is self-consistent

    try:
        AuditLog(str(path), verify_on_open=True, expected_head=expected_head)
    except AuditIntegrityError as error:
        refusal = str(error)
    else:
        raise AssertionError("truncated record unexpectedly matched the saved head")
    assert path.read_bytes() == damaged

    backup = _backup(path)
    print(f"ANCHOR REFUSED: {refusal}")
    print("The external head stays authoritative; do not lower it to the local prefix.")
    path.write_bytes(trusted)
    _verified(path, expected_head)
    resumed = AuditLog(str(path), verify_on_open=True, expected_head=expected_head)
    entry = resumed.append(compute_verdict([]), action="resumed")
    next_head = (entry["seq"] + 1, entry["hash"])
    _verified(path, next_head)
    print("Recovered the missing records, matched the saved head, then extended the chain.")
    return path, backup, expected_head, next_head


def run_examples(directory):
    """Use a new, caller-owned temporary directory, with no other log writers."""
    framing_example(directory)
    integrity_example(directory)
    anchor_example(directory)


def main():
    with TemporaryDirectory(prefix="recusal-audit-operations-") as directory:
        run_examples(Path(directory))


if __name__ == "__main__":
    main()
