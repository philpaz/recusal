"""
Offline OpenAI Agents SDK example: every gated function tool call on an audit log.

examples/openai_agents_gate.py refuses unsafe function tool calls in an OpenAI Agents SDK
agent. This example adds the record: each call a guardrail sees, allowed or refused, is one
hash-chained entry on a Recusal AuditLog, labeled with the surface
``openai_agents.tool_input_guardrail`` and the SDK's own tool call id, so a quiet edit, a
reordering, or a dropped entry is detectable with ``recusal audit verify``. The log holds a
SHA-256 fingerprint of each call's arguments, never the arguments themselves.

The audited gate is the ``recusal-openai-agents`` package (``adapters/openai-agents`` in
this repository), imported here: ``audited_agent(agent, policy,
log)`` is ``gated_agent`` and ``audited_tools(tools, policy, log)`` is ``gate_tools``, each
with a ``recording`` that passes the recording arguments through to ``recusal.decide``,
which writes the entry before returning. If the log cannot be written the decision becomes
a deny, so nothing runs unrecorded. ``Runner.run`` and ``Runner.run_streamed`` both pass
through the same guardrail.

What each entry is bound to:

- a call that is judged (allowed or refused by the policy): the invocation the SDK
  prepared, validated, defaults filled and types coerced, named by parameter and projected
  to JSON data, which is what the function receives;
- a call refused before judgment (arguments that are not a JSON object, a schema whose
  prepared arguments cannot be judged, a guardrail bound to another tool): the arguments
  as the model proposed them.

What is not on the log: a call to a tool the agent does not have (the SDK raises
``ModelBehaviorError`` before any guardrail runs, and the run stops), a call a human
approval rejected (the guardrail runs after approval, when the tool is about to execute),
and the tool calls of a sub-agent behind ``agent.as_tool(...)`` unless that sub-agent was
built with ``audited_agent`` itself.

Requires Python 3.10+ and is tested with exactly openai-agents==0.24.0 and openai==3.34.0.
Recusal itself remains Python 3.9+ and has no OpenAI dependency.

No model, API key or database is used. run_sql only records SQL text.

Run after installing:
    pip install openai-agents==0.24.0 openai==3.34.0
    python examples/openai_agents_audit.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Optional

OPENAI_AGENTS_VERSION = "0.24.0"
OPENAI_VERSION = "3.34.0"

if sys.version_info < (3, 10):
    print(
        "openai_agents_audit.py requires Python 3.10+ "
        f"(openai-agents {OPENAI_AGENTS_VERSION}); "
        f"this is Python {sys.version_info[0]}.{sys.version_info[1]}.",
        file=sys.stderr,
    )
    raise SystemExit(2)

try:
    from agents import Agent
except ImportError as exc:
    raise SystemExit(
        "Install the example dependencies first: "
        f"openai-agents=={OPENAI_AGENTS_VERSION} openai=={OPENAI_VERSION}"
    ) from exc

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
# In a checkout the adapter is read from the tree; installed, `pip install
# recusal-openai-agents` provides the same package and this path does not exist.
sys.path.insert(0, os.path.join(_ROOT, "adapters", "openai-agents", "src"))

from recusal_openai_agents import (  # noqa: E402
    SURFACE,
    Policy,
    audited_agent,
    audited_tools,
    recording,
)

from examples.openai_agents_gate import (  # noqa: E402
    EXECUTED,
    echo,
    run_agent,
    run_sql,
    scripted_model,
    sql_scope_policy,
    tool_outputs,
)
from recusal.audit import AuditLog  # noqa: E402

__all__ = ["SURFACE", "audited_agent", "audited_tools", "recording"]


def run_calls(
    calls: list[tuple[str, Any, str]],
    log: AuditLog,
    *,
    policy: Policy = sql_scope_policy,
    streamed: bool = False,
    tools: Optional[list[Any]] = None,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, str], list[tuple[str, dict[str, Any]]]]:
    """Propose the calls to an agent behind the audited gate; return outputs and what ran."""
    EXECUTED.clear()
    model = scripted_model(calls)
    agent = Agent(name="assistant", model=model, tools=[echo, run_sql] if tools is None else tools)
    agent = audited_agent(agent, policy, log, actor=actor, control=control)
    asyncio.run(run_agent(agent, streamed=streamed))
    return tool_outputs(model), list(EXECUTED)


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
        path = os.path.join(folder, "openai-agents-audit.jsonl")
        log = AuditLog(path=path)
        outputs, executed = run_calls(calls, log, actor="demo-agent")

        print("RECUSAL - OpenAI Agents SDK, every function tool call on the record (offline)\n")
        for name, _, call_id in calls:
            outcome = "refused" if outputs[call_id].startswith("Recusal") else "ran"
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
