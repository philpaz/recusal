"""
Recusal — Claude tool-call gate, offline demo (no API key, no dependencies).

Reproduces the "wrong-subject write" failure mode: a Claude agent, mid-conversation
about one customer, stages a write that targets a *different* customer. The gate
adjudicates the evidence and refuses *before the tool runs* — and hands Claude a
reason it can act on.

    python examples/claude_refusal.py
"""

import os
import sys

# Make `recusal` importable when run straight from the repo (no install needed).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recusal import Finding  # noqa: E402
from recusal.claude import gate_tool_use  # noqa: E402

# The session's verified subject this turn — the customer the user is actually asking about.
ACTIVE_CUSTOMER = {"customer_id": "C1001", "name": "Bob Smith"}


def gather_evidence(tool_input: dict) -> list:
    """Deterministic precondition: a write must target the active customer.

    This is the kind of invariant the model cannot self-enforce — it doesn't know
    which subject is 'active' in your system. The gate does.
    """
    target = tool_input.get("customer_id")
    active = ACTIVE_CUSTOMER["customer_id"]
    if target != active:
        return [
            Finding.fail(
                "subject_match",
                severity="CRITICAL",
                message=f"write targets {target} but the active customer this turn is {active}",
                target=target,
                active=active,
            )
        ]
    return [
        Finding.ok(
            "subject_match",
            severity="CRITICAL",
            message="write targets the active customer",
            target=target,
        )
    ]


def propose(tool_use_id, tool_input):
    print(f"\n  proposed: update_customer_record({_fmt(tool_input)})")
    allow, refusal = gate_tool_use(
        tool_use_id, gather_evidence(tool_input), tool_name="update_customer_record"
    )
    if allow:
        print("  gate verdict: PASS")
        print("  -> ALLOWED. The tool executes.")
    else:
        print("  gate verdict: FAIL  (CRITICAL)")
        print("  -> REFUSED before the tool ran.\n")
        print("  This is returned to Claude as a tool_result (is_error=true):")
        print(f'    "{refusal["content"]}"')
        print("\n  Claude self-corrects instead of writing to the wrong customer.")


def _fmt(d):
    return ", ".join(f'{k}="{v}"' for k, v in d.items())


def main():
    print("RECUSAL - Claude tool-call gate (offline demo)")
    print(f"Session active customer: {ACTIVE_CUSTOMER['name']} ({ACTIVE_CUSTOMER['customer_id']})")

    print("\nTurn 1 - Claude proposes a write to the WRONG customer:")
    propose("toolu_01", {"customer_id": "C-9988", "field": "loyalty_tier", "value": "Gold"})

    print("\nTurn 2 - Claude proposes the corrected call:")
    propose("toolu_02", {"customer_id": "C1001", "field": "loyalty_tier", "value": "Gold"})


if __name__ == "__main__":
    main()
