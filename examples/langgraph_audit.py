"""
Offline LangGraph example: every gated ToolNode call on a tamper-evident audit log.

examples/langgraph_gate.py refuses unsafe tool calls in a LangGraph agent. This example
adds the record: each call the agent proposes, allowed or refused, is one hash-chained
entry on a Recusal AuditLog, labeled with the surface ``langgraph.toolnode`` and LangGraph's
own tool call id, so a quiet edit, a reordering, or a dropped entry is detectable with
``recusal audit verify``. The log holds a SHA-256 fingerprint of each call's arguments,
never the arguments themselves.

The decision logic is the one in langgraph_gate.py: ``recusal_audited_gate(policy, log)``
and ``recusal_audited_agate(policy, log)`` call the same ``_judge`` with the recording
arguments passed through to ``recusal.claude_code.decide``, which writes the entry before
returning. If the log cannot be written the decision becomes a deny, so nothing runs
unrecorded.

Requires Python 3.10+ and is tested with exactly langgraph==1.2.12 and
langchain-core==1.6.5. Recusal itself remains Python 3.9+ and has no LangGraph dependency.

No LLM, API key or database is used. run_sql only records SQL text.

Run after installing:
    pip install langgraph==1.2.12 langchain-core==1.6.5
    python examples/langgraph_audit.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Optional

LANGGRAPH_VERSION = "1.2.12"
LANGCHAIN_CORE_VERSION = "1.6.5"

if sys.version_info < (3, 10):
    print(
        "langgraph_audit.py requires Python 3.10+ "
        f"(LangGraph {LANGGRAPH_VERSION}); "
        f"this is Python {sys.version_info[0]}.{sys.version_info[1]}.",
        file=sys.stderr,
    )
    raise SystemExit(2)

try:
    from langgraph.prebuilt import ToolNode
except ImportError as exc:
    raise SystemExit(
        "Install the example dependencies first: "
        f"langgraph=={LANGGRAPH_VERSION} langchain-core=={LANGCHAIN_CORE_VERSION}"
    ) from exc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.langgraph_gate import (  # noqa: E402
    EXECUTED,
    Policy,
    _judge,
    echo,
    run_in_graph,
    run_in_graph_async,
    run_sql,
    sql_scope_policy,
)
from recusal.audit import AuditLog  # noqa: E402

#: The surface label every entry carries: the runtime that adjudicated, LangGraph's ToolNode.
SURFACE = "langgraph.toolnode"


def _recording(request, actor: Optional[str], control: Optional[dict[str, Any]], log: AuditLog):
    """The recording arguments for one call, built in one place for both gates."""
    return {
        "audit": log,
        "surface": SURFACE,
        "tool_use_id": request.tool_call["id"],
        "actor": actor,
        "control": control,
    }


def recusal_audited_gate(
    policy: Policy,
    log: AuditLog,
    *,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
):
    """Return a synchronous ToolNode hook that records, then gates, each call."""

    def gate(request, execute):
        refusal, judged = _judge(request, policy, **_recording(request, actor, control, log))
        return refusal if refusal is not None else execute(judged)

    return gate


def recusal_audited_agate(
    policy: Policy,
    log: AuditLog,
    *,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
):
    """Return an asynchronous ToolNode hook for awrap_tool_call, same logic as the sync one."""

    async def gate(request, execute):
        refusal, judged = _judge(request, policy, **_recording(request, actor, control, log))
        return refusal if refusal is not None else await execute(judged)

    return gate


def run_calls(
    calls: list[dict[str, Any]],
    log: AuditLog,
    *,
    policy: Policy = sql_scope_policy,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
):
    """Run the scripted graph with the audited sync gate; return (messages, executed)."""
    EXECUTED.clear()
    node = ToolNode(
        [echo, run_sql],
        wrap_tool_call=recusal_audited_gate(policy, log, actor=actor, control=control),
    )
    return run_in_graph(node, calls), list(EXECUTED)


async def run_calls_async(
    calls: list[dict[str, Any]],
    log: AuditLog,
    *,
    policy: Policy = sql_scope_policy,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
):
    """Run the scripted graph with the audited async gate; return (messages, executed)."""
    EXECUTED.clear()
    node = ToolNode(
        [echo, run_sql],
        awrap_tool_call=recusal_audited_agate(policy, log, actor=actor, control=control),
    )
    return await run_in_graph_async(node, calls), list(EXECUTED)


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
        {"name": "echo", "args": {"text": "hello"}, "id": "c1", "type": "tool_call"},
        {
            "name": "run_sql",
            "args": {"sql": "DELETE FROM audit_log"},
            "id": "c2",
            "type": "tool_call",
        },
        {
            "name": "run_sql",
            "args": {"sql": "SELECT * FROM audit_log"},
            "id": "c3",
            "type": "tool_call",
        },
    ]
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "langgraph-audit.jsonl")
        log = AuditLog(path=path)
        messages, executed = run_calls(calls, log, actor="demo-agent")

        print("RECUSAL - LangGraph ToolNode, every call on the record (offline)\n")
        for message in messages:
            print(f"  {message.tool_call_id} {message.name:<8} {message.status}")
        print(f"  executed: {', '.join(name for name, _ in executed)}\n")

        # ToolNode runs one message's calls in parallel: entries are in completion order,
        # so read them by tool_use_id, never by position.
        by_id = {entry["action"]["tool_use_id"]: entry for entry in log.entries}
        for call in calls:
            entry = by_id[call["id"]]
            action = entry["action"]
            print(
                f"  {call['id']} {action['surface']} {action['tool']:<8} "
                f"{action['decision']:<5} {entry['actor']} {entry['hash'][:12]}..."
            )

        code, text = verify_with_cli(path)
        print(f"\n  verify intact:   exit {code}  {text}")

        tamper(path, "c2")  # the refused DELETE now reads as allowed
        code, text = verify_with_cli(path)
        print(f"  verify tampered: exit {code}  {text}")


if __name__ == "__main__":
    main()
