"""
Offline LangChain create_agent example: every gated tool call on an audit log.

examples/langchain_agent_gate.py gates a LangChain ``create_agent`` agent with the shared
ToolNode gate as middleware. This example adds the record: each call the agent proposes,
allowed or refused, is one hash-chained entry on a Recusal AuditLog, labeled with the
surface ``langchain.create_agent`` and LangChain's own tool call id, so a quiet edit, a
reordering, or a dropped entry is detectable with ``recusal audit verify``. The log holds a
SHA-256 fingerprint of each call's arguments, never the arguments themselves.

The audited middleware is the ``recusal-langchain`` package (``adapters/langchain`` in
this repository), imported here. Its decision logic is the LangGraph gate's:
``audited_gate_middleware(policy, log)`` and ``audited_agate_middleware(policy, log)``
wrap the audited ToolNode hooks of ``recusal-langgraph``, which call the plain gate's own
judging function with the recording arguments passed through to ``recusal.decide``. If
the log cannot be written the
decision becomes a deny, so nothing runs unrecorded. Each entry is bound to the invocation
the tool runs with, as the LangGraph audited gate binds it.

List the middleware LAST, as with the plain gate. The gate judges, and records, inside the
tool's own lifecycle, so an argument rewrite by a middleware listed after it is still
judged and recorded as rewritten; but a middleware listed after it that swaps the tool
object, or answers in the tool's place, removes the judging copy, and that call is neither
judged nor recorded.

Requires Python 3.10+, langgraph==1.2.12, langchain-core==1.6.5 and langchain==1.4.3.
Recusal itself remains Python 3.9+ and has no LangChain dependency. No model, credential
or database is contacted.

Run after installing:
    pip install langgraph==1.2.12 langchain-core==1.6.5 langchain==1.4.3
    python examples/langchain_agent_audit.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any, Optional

LANGGRAPH_VERSION = "1.2.12"
LANGCHAIN_CORE_VERSION = "1.6.5"
LANGCHAIN_VERSION = "1.4.3"

if sys.version_info < (3, 10):
    print(
        "langchain_agent_audit.py requires Python 3.10+ "
        f"(LangChain {LANGCHAIN_VERSION}); "
        f"this is Python {sys.version_info[0]}.{sys.version_info[1]}.",
        file=sys.stderr,
    )
    raise SystemExit(2)

try:
    from langchain_core.messages import HumanMessage, ToolMessage
except ImportError as exc:
    raise SystemExit(
        "Install the example dependencies first: "
        f"langgraph=={LANGGRAPH_VERSION} langchain-core=={LANGCHAIN_CORE_VERSION} "
        f"langchain=={LANGCHAIN_VERSION}"
    ) from exc

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
# In a checkout the adapters are read from the tree; installed, `pip install
# recusal-langchain` provides the same packages and these paths do not exist.
sys.path.insert(0, os.path.join(_ROOT, "adapters", "langgraph", "src"))
sys.path.insert(0, os.path.join(_ROOT, "adapters", "langchain", "src"))

from recusal_langchain import (  # noqa: E402
    SURFACE,
    audited_agate_middleware,
    audited_gate_middleware,
)

from examples.langchain_agent_gate import (  # noqa: E402
    EXECUTED,
    scripted_agent,
    sql_scope_policy,
    tool_messages,
)
from examples.langgraph_audit import (  # noqa: E402
    tamper,
    verify_with_cli,
)
from recusal.audit import AuditLog  # noqa: E402

__all__ = [
    "SURFACE",
    "AuditLog",
    "audited_agate_middleware",
    "audited_gate_middleware",
    "tamper",
    "verify_with_cli",
]


def run_calls(
    calls: list[dict[str, Any]],
    log: AuditLog,
    *,
    policy=sql_scope_policy,
    tools: Optional[list[Any]] = None,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
) -> tuple[list[ToolMessage], list[tuple[str, dict[str, Any]]]]:
    """Run fake model proposals through create_agent behind the audited gate."""
    EXECUTED.clear()
    middleware = [audited_gate_middleware(policy, log, actor=actor, control=control)]
    agent = scripted_agent(calls, middleware, tools)
    result = agent.invoke({"messages": [HumanMessage(content="run the scripted tools")]})
    return tool_messages(result), list(EXECUTED)


async def arun_calls(
    calls: list[dict[str, Any]],
    log: AuditLog,
    *,
    policy=sql_scope_policy,
    tools: Optional[list[Any]] = None,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
) -> tuple[list[ToolMessage], list[tuple[str, dict[str, Any]]]]:
    """The same run through ``agent.ainvoke`` behind the audited async gate."""
    EXECUTED.clear()
    middleware = [audited_agate_middleware(policy, log, actor=actor, control=control)]
    agent = scripted_agent(calls, middleware, tools)
    result = await agent.ainvoke({"messages": [HumanMessage(content="run the scripted tools")]})
    return tool_messages(result), list(EXECUTED)


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
        path = os.path.join(folder, "langchain-agent-audit.jsonl")
        log = AuditLog(path=path)
        messages, executed = run_calls(calls, log, actor="demo-agent")

        print("RECUSAL - LangChain create_agent, every call on the record (offline)\n")
        for message in messages:
            print(f"  {message.tool_call_id} {message.name:<8} {message.status}")
        print(f"  executed: {', '.join(name for name, _ in executed)}\n")

        # One message's calls run in parallel: read entries by tool_use_id, never by position.
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
