"""
Recusal — tamper-evident audit log demo (offline, no API key).

Records three verdicts, verifies the chain is intact, then quietly edits one
record and shows the chain detect it.

    python examples/audit_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone  # noqa: E402

from recusal import compute_verdict  # noqa: E402
from recusal.audit import AuditLog, verify  # noqa: E402

FIXED = datetime(2026, 1, 1, tzinfo=timezone.utc)


def main():
    log = AuditLog(clock=lambda: FIXED)
    log.append(
        compute_verdict([{"severity": "CRITICAL", "status": "fail", "message": "rm -rf refused"}]),
        action={"tool": "Bash", "command": "rm -rf /"},
        actor="agent-1",
    )
    log.append(compute_verdict([]), action={"tool": "Read", "file": "README.md"}, actor="agent-1")
    log.append(
        compute_verdict([{"severity": "ERROR", "status": "fail", "message": "coverage 61% < 75%"}]),
        action={"tool": "merge_pr"},
        actor="agent-1",
    )

    print("RECUSAL - tamper-evident audit log (offline)\n")
    for e in log.entries:
        print(
            f"  #{e['seq']} {e['decision']:<5} {str(e['action']):<42} "
            f"hash={e['hash'][:12]} prev={e['prev_hash'][:12]}"
        )

    ok, _ = verify(log.entries)
    print(f"\n  verify: {'INTACT' if ok else 'TAMPERED'}")

    print("\n  ...now an auditor's record is quietly edited:")
    log.entries[0]["reasons"] = "(nothing happened here)"
    ok, problems = verify(log.entries)
    print(f"  verify: {'INTACT' if ok else 'TAMPERED'}")
    for p in problems:
        print(f"    - {p}")


if __name__ == "__main__":
    main()
