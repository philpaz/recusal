"""Mirror audit entries and check a saved head before resuming (offline).

Run: python examples/audit_sink.py

This local second file demonstrates the AuditSink seam, not an external anchor.
An attacker able to rewrite both files can defeat the check. A real anchor must
live somewhere the attacker cannot reach. This single-process example keeps its
head in memory; production storage must also persist and protect that head across
restarts and provide its own durability and concurrency guarantees.
"""

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recusal import Finding, compute_verdict
from recusal.audit import GENESIS, AuditIntegrityError, AuditLog, verify_file


class MirrorSink:
    """A structural sink: only write(entry) is required, with no base class.

    Exceptions propagate to the caller. The local audit entry is already committed
    when write runs, so a failed sink delivery does not roll it back. Sequence gaps
    are possible on retries; the head comes from the entry, not a local counter.
    """

    def __init__(self, path: Path):
        self.path = path
        self.head = (0, GENESIS)

    def write(self, entry):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=True) + "\n")
        self.head = (entry["seq"] + 1, entry["hash"])


def main():
    with TemporaryDirectory(prefix="recusal-audit-sink-") as directory:
        path = Path(directory) / "audit.jsonl"
        sink = MirrorSink(Path(directory) / "mirror.jsonl")
        verdict = compute_verdict([Finding.ok("example")])
        log = AuditLog(str(path), sinks=[sink])
        log.append(verdict, action="first action")

        resumed = AuditLog(str(path), sinks=[sink], verify_on_open=True, expected_head=sink.head)
        resumed.append(verdict, action="second action")
        assert path.read_text(encoding="utf-8") == sink.path.read_text(encoding="utf-8")
        intact, problems = verify_file(str(sink.path), expected_head=sink.head)
        assert intact, problems
        print("Mirrored 2 entries; verified resume extended the same chain.")
        print(f"Saved head: count={sink.head[0]}, hash={sink.head[1]}")

        # The remaining prefix is valid on its own; only the saved head reveals loss.
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        path.write_text(first_line + "\n", encoding="utf-8")
        assert verify_file(str(path))[0]
        try:
            AuditLog(str(path), verify_on_open=True, expected_head=sink.head)
        except AuditIntegrityError:
            print("REFUSED: truncated local log does not match the saved head.")
        else:
            raise AssertionError("Truncated log unexpectedly accepted")
        print("Local mirror only: a real external anchor must be beyond the attacker's reach.")


if __name__ == "__main__":
    main()
