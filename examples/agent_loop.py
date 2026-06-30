"""
Framework-neutral agent loop — the zero-dependency core gating tool calls with no
Claude, no SDK, no third-party anything. This is the proof behind the claim that
Recusal works in *any* agent loop: the only import is ``recusal`` itself, and the
loop below is a plain ``for`` over proposed actions. Swap it for LangGraph, the
OpenAI Agents SDK, a homegrown runtime — the gate is identical.

    python examples/agent_loop.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recusal import Finding, compute_verdict  # noqa: E402

# --- your policy: a proposed tool call -> evidence findings. No model in here. -------
ALLOWED_TABLES = {"audit_log", "staging"}


def evaluate(tool: str, args: dict) -> list:
    """Turn a proposed (tool, args) call into findings. This is the whole policy —
    deterministic, yours to own, and the only thing the verdict depends on."""
    findings = []
    if tool == "shell" and "rm -rf" in args.get("cmd", ""):
        findings.append(
            Finding.fail("destructive_shell", severity="CRITICAL", message="refusing rm -rf")
        )
    if tool == "sql":
        sql = args.get("sql", "").lower()
        if "delete" in sql and "where" not in sql:
            findings.append(
                Finding.fail("unscoped_delete", severity="CRITICAL", message="DELETE without WHERE")
            )
    if tool == "write_table" and args.get("table") not in ALLOWED_TABLES:
        findings.append(
            Finding.fail(
                "table_allowlist",
                severity="ERROR",
                message=f"table '{args.get('table')}' not in allowlist",
            )
        )
    return findings


# --- a generic loop: propose -> gate -> act / refuse. The gate is the only seam. ----
def run(proposed: list) -> None:
    for tool, args in proposed:
        verdict = compute_verdict(evaluate(tool, args))
        if verdict.refused:
            print(f"  REFUSE  {tool}({args})  ->  {verdict.reasons()}")
        elif verdict.retryable:
            print(f"  RETRY   {tool}({args})  ->  {verdict.reasons()}")
        else:
            print(f"  ALLOW   {tool}({args})")
            # execute_tool(tool, args)   # <- the real, irreversible action goes here


PROPOSED = [
    ("sql", {"sql": "DELETE FROM orders"}),  # unscoped       -> REFUSE
    ("shell", {"cmd": "rm -rf /data"}),  # destructive    -> REFUSE
    ("write_table", {"table": "customers"}),  # not allowlisted -> RETRY
    ("write_table", {"table": "audit_log"}),  # allowed         -> ALLOW
    ("sql", {"sql": "DELETE FROM orders WHERE id = 7"}),  # scoped  -> ALLOW
]


def main() -> None:
    print("RECUSAL - framework-neutral agent loop (no Claude, no SDK)\n")
    run(PROPOSED)
    print("\n  Only import: `recusal`. The same gate drops into any agent runtime.")


if __name__ == "__main__":
    main()
