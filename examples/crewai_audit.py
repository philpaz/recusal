"""
Offline CrewAI example: every gated tool call on a tamper-evident audit log.

examples/crewai_gate.py refuses unsafe tool calls in a CrewAI crew through CrewAI's own
tool-call hooks. This example adds the record: each call the before hook sees, allowed or
refused, is one hash-chained entry on a Recusal AuditLog, labeled with the surface
``crewai.before_tool_call``, so a quiet edit, a reordering, or a dropped entry is detectable
with ``recusal audit verify``. The log holds a SHA-256 fingerprint of each call's arguments,
never the arguments themselves.

The decision logic is the one in crewai_gate.py: ``install_audited_hooks(policy, log)`` is
``install_recusal_hooks`` with a ``recording`` that passes the recording arguments through
to ``recusal.decide``, which writes the entry before returning. If the log cannot be written
the decision becomes a deny, so nothing runs unrecorded. Like the plain gate it is
process-global: one install records every agent, task and crew in the process.

What each entry carries:

- no call id. CrewAI's hook context has the tool name, the argument dict, the tool and the
  agent, task and crew, and no id for the call, so the record does not invent one. Entries
  are told apart by their order on the chain, the tool, the fingerprint and
  ``control.crewai_agent``, the role of the agent that made the call when CrewAI supplies
  it;
- for a judged call (allowed or refused by the policy): the fingerprint of the prepared
  invocation CrewAI executes, defaults completed and values validated;
- for a call refused before judgment (a tool or schema the gate cannot judge, an unknown or
  missing parameter, a validation failure): the fingerprint of the arguments as proposed.

What is not on the log: a call a hook registered BEFORE the gate blocked (CrewAI stops at
the first hook that blocks, so the gate never sees it and the tool does not run), and
whatever a hook registered AFTER the gate changed. Install the gate after your other hooks,
as with the plain one, so the record is of the call exactly as it runs.

Requires Python 3.10+ (CrewAI's minimum) and is tested with exactly crewai==1.15.34.
Recusal itself remains Python 3.9+ and has no CrewAI dependency. No model, API key or
network is used, and run_sql only records SQL text.

Run after installing:
    pip install crewai==1.15.34
    python examples/crewai_audit.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Callable, Optional

CREWAI_VERSION = "1.15.34"

if sys.version_info < (3, 10):
    print(
        "crewai_audit.py requires Python 3.10+ "
        f"(crewai {CREWAI_VERSION}); "
        f"this is Python {sys.version_info[0]}.{sys.version_info[1]}.",
        file=sys.stderr,
    )
    raise SystemExit(2)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.crewai_gate import (  # noqa: E402
    EXECUTED,
    Policy,
    Recording,
    install_recusal_hooks,
    run_crew,
    scripted_agent,
    sql_scope_policy,
)
from recusal.audit import AuditLog  # noqa: E402

#: The surface label every entry carries: the CrewAI hook that adjudicated.
SURFACE = "crewai.before_tool_call"


def recording(
    log: AuditLog, *, actor: Optional[str] = None, control: Optional[dict[str, Any]] = None
) -> Recording:
    """The recording arguments for each call, built in one place.

    CrewAI's hook context carries no call id, so none is recorded. The role of the agent
    making the call is recorded as ``control.crewai_agent`` when CrewAI supplies one; a
    caller's own ``control`` cannot overwrite it."""

    def for_call(context: Any) -> dict[str, Any]:
        identity = dict(control or {})
        role = getattr(getattr(context, "agent", None), "role", None)
        if isinstance(role, str) and role:
            identity["crewai_agent"] = role
        else:
            identity.pop("crewai_agent", None)
        return {
            "audit": log,
            "surface": SURFACE,
            "actor": actor,
            "control": identity or None,
        }

    return for_call


def install_audited_hooks(
    policy: Policy,
    log: AuditLog,
    *,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
) -> Callable[[], None]:
    """``install_recusal_hooks`` with every decision on ``log``; returns ``uninstall``."""
    return install_recusal_hooks(policy, recording(log, actor=actor, control=control))


def run_calls(
    calls: list[tuple[str, Any, str]],
    log: AuditLog,
    *,
    policy: Policy = sql_scope_policy,
    tools: Optional[list] = None,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, str], list[tuple[str, dict[str, Any]]]]:
    """Propose the calls to a crew behind the audited gate; return answers and what ran."""
    EXECUTED.clear()
    uninstall = install_audited_hooks(policy, log, actor=actor, control=control)
    try:
        responses = run_crew(scripted_agent(calls, tools=tools))
    finally:
        uninstall()
    return responses, list(EXECUTED)


def verify_with_cli(path: str) -> tuple[int, str]:
    """Run ``recusal audit verify`` on the log the way an auditor would; (exit code, text)."""
    result = subprocess.run(
        [sys.executable, "-m", "recusal", "audit", "verify", path],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, (result.stdout + result.stderr).strip()


def tamper(path: str, seq: int, decision: str = "allow") -> None:
    """Rewrite one entry's recorded decision in place, the quiet edit the chain must catch.
    By sequence number: these entries carry no call id."""
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    for index, line in enumerate(lines):
        entry = json.loads(line)
        if entry["seq"] == seq:
            entry["action"]["decision"] = decision
            lines[index] = json.dumps(entry, sort_keys=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    calls = [
        ("echo", {"text": "hello"}, "c1"),
        ("run_sql", {"sql": "DELETE FROM audit_log"}, "c2"),
        ("run_sql", {"sql": "SELECT * FROM audit_log"}, "c3"),
    ]
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "crewai-audit.jsonl")
        log = AuditLog(path=path)
        responses, executed = run_calls(calls, log, actor="demo-agent")

        print("RECUSAL - CrewAI crew, every tool call on the record (offline)\n")
        for name, _, call_id in calls:
            outcome = "refused" if responses[call_id].startswith("Recusal") else "ran"
            print(f"  {call_id} {name:<8} {outcome}")
        print(f"  executed: {', '.join(name for name, _ in executed)}\n")

        for entry in log.entries:
            action = entry["action"]
            print(
                f"  {entry['seq']} {action['surface']} {action['tool']:<8} "
                f"{action['decision']:<5} {entry['actor']} "
                f"agent={action['control'].get('crewai_agent', '-')} {entry['hash'][:12]}..."
            )

        code, text = verify_with_cli(path)
        print(f"\n  verify intact:   exit {code}  {text}")

        refused = next(e["seq"] for e in log.entries if e["action"]["decision"] == "deny")
        tamper(path, refused)  # the refused DELETE now reads as allowed
        code, text = verify_with_cli(path)
        print(f"  verify tampered: exit {code}  {text}")


if __name__ == "__main__":
    main()
