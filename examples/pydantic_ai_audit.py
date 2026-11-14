"""
Offline Pydantic AI example: every gated tool call on a tamper-evident audit log.

examples/pydantic_ai_gate.py refuses unsafe tool calls in a Pydantic AI agent. This example
adds the record: each call the gate sees, allowed or refused, is one hash-chained entry on
a Recusal AuditLog, labeled with the surface ``pydantic_ai.before_tool_execute`` and
Pydantic AI's own tool call id, so a quiet edit, a reordering, or a dropped entry is
detectable with ``recusal audit verify``. The log holds a SHA-256 fingerprint of each
call's arguments, never the arguments themselves.

The audited gate is the ``recusal-pydantic-ai`` package (``adapters/pydantic-ai`` in this
repository): ``RecusalAuditedGate(policy, log=log)`` and
``recusal_audited_before_tool_execute(policy, log)`` are imported from it. They call the
plain gate's own refusal function with the recording arguments passed through to
``recusal.decide``, which writes the entry before returning. If the log cannot be written
the decision becomes a deny, so nothing runs unrecorded.

What each entry is bound to:

- a call that is judged (allowed or refused by the policy): the validated arguments the
  tool is called with;
- a call refused before judgment (arguments the gate cannot detach): the call as the model
  proposed it, since the validated arguments are not data a fingerprint can cover.

What is not on the log, because the hook never runs for it: a call whose arguments fail
Pydantic AI's validation (answered with a retry prompt, and the tool does not run), and
the tool calls of an agent another agent delegates to, unless that agent has its own
audited gate.

Requires Python 3.11+ (Pydantic AI's minimum) and is tested with exactly
pydantic-ai-slim==2.64.0. Recusal itself remains Python 3.9+ and has no Pydantic AI
dependency.

No model, API key or database is used. run_sql only records SQL text.

Run after installing:
    pip install pydantic-ai-slim==2.64.0
    python examples/pydantic_ai_audit.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Optional

PYDANTIC_AI_VERSION = "2.64.0"

if sys.version_info < (3, 11):
    print(
        "pydantic_ai_audit.py requires Python 3.11+ "
        f"(pydantic-ai-slim {PYDANTIC_AI_VERSION}); "
        f"this is Python {sys.version_info[0]}.{sys.version_info[1]}.",
        file=sys.stderr,
    )
    raise SystemExit(2)

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

try:
    from pydantic_ai.capabilities import Hooks
except ImportError as exc:
    raise SystemExit(
        f"Install the example dependency first: pydantic-ai-slim=={PYDANTIC_AI_VERSION}"
    ) from exc

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
# In a checkout the adapter is read from the tree; installed, `pip install
# recusal-pydantic-ai` provides the same package and this path does not exist.
sys.path.insert(0, os.path.join(_ROOT, "adapters", "pydantic-ai", "src"))

from recusal_pydantic_ai import (  # noqa: E402
    SURFACE,
    Policy,
    RecusalAuditedGate,
    recusal_audited_before_tool_execute,
)

from examples.pydantic_ai_gate import (  # noqa: E402
    EXECUTED,
    responses_of,
    scripted_agent,
    sql_scope_policy,
)
from recusal.audit import AuditLog  # noqa: E402

__all__ = ["SURFACE", "RecusalAuditedGate", "recusal_audited_before_tool_execute"]


def run_calls(
    calls: list[tuple[str, Any, str]],
    log: AuditLog,
    *,
    policy: Policy = sql_scope_policy,
    via_hooks: bool = False,
    tools: Optional[list] = None,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, tuple[str, Any]], list[tuple[str, dict[str, Any]]]]:
    """Propose the calls to an agent behind the audited gate; return answers and what ran."""
    EXECUTED.clear()
    if via_hooks:
        hook = recusal_audited_before_tool_execute(policy, log, actor=actor, control=control)
        capabilities: list = [Hooks(before_tool_execute=hook)]
    else:
        capabilities = [RecusalAuditedGate(policy, log=log, actor=actor, control=control)]
    result = scripted_agent(calls, tools=tools, capabilities=capabilities).run_sync("go")
    return responses_of(result), list(EXECUTED)


def verify_with_cli(path: str) -> tuple[int, str]:
    """Run ``recusal audit verify`` on the log the way an auditor would; (exit code, text)."""
    result = subprocess.run(
        [sys.executable, "-m", "recusal", "audit", "verify", path],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, (result.stdout + result.stderr).strip()


def tamper(path: str, tool_use_id: str, decision: str = "allow") -> None:
    """Rewrite one entry's recorded decision in place, the quiet edit the chain must catch."""
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    for index, line in enumerate(lines):
        entry = json.loads(line)
        if entry["action"].get("tool_use_id") == tool_use_id:
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
        path = os.path.join(folder, "pydantic-ai-audit.jsonl")
        log = AuditLog(path=path)
        responses, executed = run_calls(calls, log, actor="demo-agent")

        print("RECUSAL - Pydantic AI agent, every call on the record (offline)\n")
        for name, _, call_id in calls:
            print(f"  {call_id} {name:<8} {responses[call_id][0]}")
        print(f"  executed: {', '.join(name for name, _ in executed)}\n")

        # One model turn's calls can complete in any order: read entries by tool_use_id,
        # never by position.
        by_id = {entry["action"]["tool_use_id"]: entry for entry in log.entries}
        for name, _, call_id in calls:
            entry = by_id[call_id]
            action = entry["action"]
            print(
                f"  {call_id} {action['surface']} {action['tool']:<8} "
                f"{action['decision']:<5} {entry['actor']} {entry['hash'][:12]}..."
            )

        code, text = verify_with_cli(path)
        print(f"\n  verify intact:   exit {code}  {text}")

        tamper(path, "c2")  # the refused DELETE now reads as allowed
        code, text = verify_with_cli(path)
        print(f"  verify tampered: exit {code}  {text}")


if __name__ == "__main__":
    main()
