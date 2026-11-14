"""
Offline Google ADK example: every gated tool call on a tamper-evident audit log.

examples/google_adk_gate.py refuses unsafe tool calls in a Google ADK agent. This example
adds the record: each call a gate sees, allowed or refused, is one hash-chained entry on a
Recusal AuditLog, labeled with the surface ``google_adk.before_tool_callback`` and the ADK's
own function call id, so a quiet edit, a reordering, or a dropped entry is detectable with
``recusal audit verify``. The log holds a SHA-256 fingerprint of each call's arguments,
never the arguments themselves.

The audited gate is the ``recusal-google-adk`` package (``adapters/google-adk`` in this
repository), imported here: ``audited_agent(agent, policy, log)``
is ``gated_agent`` and ``audited_plugin(policy, log)`` is ``RecusalPlugin``, each with a
``recording`` that passes the recording arguments through to ``recusal.decide``, which
writes the entry before returning. If the log cannot be written the decision becomes a
deny, so nothing runs unrecorded. ``audited_agent`` records the sub-agents and agent tools
it gates on the same log.

What each entry is bound to:

- a call that is judged (allowed or refused by the policy): the prepared invocation the
  tool is called with, defaults applied, so a default the model omitted is on the record at
  the value that runs;
- a call refused before judgment (a tool that is not a stock ``FunctionTool``, a signature
  that cannot be judged, arguments that do not validate, a plugin that is not the last
  before-tool seam): the arguments as the model proposed them.

What is not on the log: a call another before-tool callback answered before the gate ran
(the tool did not run, and the gate never saw the call), and anything ``gated_agent``
refuses to gate at construction, which raises before any agent runs.

Requires Python 3.11+ (the ADK's minimum) and is tested with exactly google-adk==2.13.0 and
google-genai==2.34.0. Recusal itself remains Python 3.9+ and has no ADK dependency.

No model, API key or database is used. run_sql only records SQL text.

Run after installing:
    pip install google-adk==2.13.0 google-genai==2.34.0
    python examples/google_adk_audit.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Optional

GOOGLE_ADK_VERSION = "2.13.0"
GOOGLE_GENAI_VERSION = "2.34.0"

if sys.version_info < (3, 11):
    print(
        "google_adk_audit.py requires Python 3.11+ "
        f"(google-adk {GOOGLE_ADK_VERSION}); "
        f"this is Python {sys.version_info[0]}.{sys.version_info[1]}.",
        file=sys.stderr,
    )
    raise SystemExit(2)

try:
    from google.adk.agents import BaseAgent
except ImportError as exc:
    raise SystemExit(
        "Install the example dependencies first: "
        f"google-adk=={GOOGLE_ADK_VERSION} google-genai=={GOOGLE_GENAI_VERSION}"
    ) from exc

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
# In a checkout the adapter is read from the tree; installed, `pip install
# recusal-google-adk` provides the same package and this path does not exist.
sys.path.insert(0, os.path.join(_ROOT, "adapters", "google-adk", "src"))

from recusal_google_adk import (  # noqa: E402
    SURFACE,
    Policy,
    audited_agent,
    audited_plugin,
    recording,
)

from examples.google_adk_gate import (  # noqa: E402
    EXECUTED,
    run_agent,
    scripted_agent,
    sql_scope_policy,
)
from recusal.audit import AuditLog  # noqa: E402

__all__ = ["SURFACE", "audited_agent", "audited_plugin", "recording"]


def run_calls(
    calls: list[tuple[str, Any, str]],
    log: AuditLog,
    *,
    policy: Policy = sql_scope_policy,
    via_plugin: bool = False,
    tools: Optional[list] = None,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    """Propose the calls to an agent behind the audited gate; return responses and what ran."""
    EXECUTED.clear()
    agent: BaseAgent = scripted_agent(calls, tools=tools)
    plugins = None
    if via_plugin:
        plugins = [audited_plugin(policy, log, actor=actor, control=control)]
    else:
        agent = audited_agent(agent, policy, log, actor=actor, control=control)
    responses = asyncio.run(run_agent(agent, plugins=plugins))
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
        path = os.path.join(folder, "google-adk-audit.jsonl")
        log = AuditLog(path=path)
        responses, executed = run_calls(calls, log, actor="demo-agent")

        print("RECUSAL - Google ADK agent, every call on the record (offline)\n")
        for name, _, call_id in calls:
            outcome = "refused" if "error" in responses[call_id] else "ran"
            print(f"  {call_id} {name:<8} {outcome}")
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
