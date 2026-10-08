"""Recipe 4: refuse writes if there is no valid active subject.

Run: python examples/subject_guard.py

This is a tool-input guard, not independent authority to perform a write.
A clean request defers to the host's existing permissions.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recusal import Finding
from recusal.claude_code import decide


def make_subject_guard(active_id):
    """Bind the trusted active subject for one session or turn."""

    def policy(tool_name, tool_input):
        if tool_name != "update_record":
            return []
        if not isinstance(active_id, str) or not active_id.strip():
            return [
                Finding.fail(
                    "subject_unbound",
                    severity="CRITICAL",
                    message="no active subject is bound; write refused",
                )
            ]
        target = tool_input.get("id")
        if not isinstance(target, str) or not target.strip() or target != active_id:
            return [
                Finding.fail(
                    "subject_match",
                    severity="CRITICAL",
                    message=f"write targets {target!r}, not the active subject {active_id!r}",
                )
            ]
        return []

    return policy


def main():
    for name, active_id, target in (
        ("same customer", "C1001", "C1001"),
        ("wrong customer", "C1001", "C2002"),
        ("unbound session", None, None),
        ("empty session", "", ""),
    ):
        decision, _reason = decide("update_record", {"id": target}, make_subject_guard(active_id))
        print(f"{name}: {decision}")


if __name__ == "__main__":
    main()
