"""
Offline LangGraph example: gate @tool calls with Recusal at ToolNode execution time.

Requires Python 3.10+ and is tested with exactly langgraph==1.2.12 and
langchain-core==1.6.5. Recusal itself remains Python 3.9+ and has no LangGraph dependency.

The hook is LangGraph-native: a user's existing @tool list goes unchanged into ToolNode and
recusal_gate(policy) is attached with wrap_tool_call. Every call in a model message crosses
the gate separately immediately before execution. Denials become error ToolMessages carrying
Recusal's reason; allow/defer execute normally. Async graphs can attach the same policy shape
through awrap_tool_call (this example stays synchronous).

No LLM, API key or database is used. run_sql only records SQL text.

Run after installing:
    pip install langgraph==1.2.12 langchain-core==1.6.5
    python examples/langgraph_gate.py
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, List

LANGGRAPH_VERSION = "1.2.12"
LANGCHAIN_CORE_VERSION = "1.6.5"

if sys.version_info < (3, 10):
    print(
        "langgraph_gate.py requires Python 3.10+ "
        f"(LangGraph {LANGGRAPH_VERSION}); "
        f"this is Python {sys.version_info[0]}.{sys.version_info[1]}.",
        file=sys.stderr,
    )
    raise SystemExit(2)

try:
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.tools import tool
    from langgraph.prebuilt import ToolNode
except ImportError as exc:
    raise SystemExit(
        "Install the example dependencies first: "
        f"langgraph=={LANGGRAPH_VERSION} langchain-core=={LANGCHAIN_CORE_VERSION}"
    ) from exc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.sql_scope_policy import policy as sql_scope_policy  # noqa: E402
from recusal.claude_code import decide  # noqa: E402

Policy = Callable[[str, dict], List[Any]]
EXECUTED: list[tuple[str, dict[str, Any]]] = []


@tool
def echo(text: str) -> str:
    """Return harmless text."""
    EXECUTED.append(("echo", {"text": text}))
    return text


@tool
def run_sql(sql: str) -> str:
    """Record SQL text without connecting to a database."""
    EXECUTED.append(("run_sql", {"sql": sql}))
    return f"recorded SQL: {sql}"


def recusal_gate(policy: Policy):
    """Return a LangGraph ToolNode hook that gates one tool call immediately before execution."""

    def gate(request, execute):
        call = request.tool_call
        try:
            decision, reason = decide(call["name"], call["args"], policy)
        except Exception as exc:
            decision, reason = "deny", f"Recusal failed closed (policy error): {exc}"
        if decision == "deny":
            return ToolMessage(
                content=reason,
                tool_call_id=call["id"],
                name=call["name"],
                status="error",
            )
        return execute(request)

    return gate


def tool_message(calls: list[dict[str, Any]]) -> AIMessage:
    """A scripted model message proposing tool calls; no model is invoked."""
    return AIMessage(content="", tool_calls=calls)


def run_calls(
    calls: list[dict[str, Any]], *, policy: Policy = sql_scope_policy, gated: bool = True
) -> tuple[list[ToolMessage], list[tuple[str, dict[str, Any]]]]:
    EXECUTED.clear()
    node = (
        ToolNode([echo, run_sql], wrap_tool_call=recusal_gate(policy))
        if gated
        else ToolNode([echo, run_sql])
    )
    result = node.invoke({"messages": [tool_message(calls)]})
    return result["messages"], list(EXECUTED)


def main() -> None:
    calls = [
        {"name": "echo", "args": {"text": "hello"}, "id": "1", "type": "tool_call"},
        {"name": "run_sql", "args": {"sql": "DELETE FROM audit_log"}, "id": "2", "type": "tool_call"},
        {"name": "run_sql", "args": {"sql": "SELECT * FROM audit_log"}, "id": "3", "type": "tool_call"},
    ]
    messages, executed = run_calls(calls)
    for message in messages:
        print(f"{message.name} [{message.status}]: {message.content}")
    print(f"tools that ran: {len(executed)}")


if __name__ == "__main__":
    main()
