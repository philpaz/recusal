"""
Offline LangGraph example: gate @tool calls with Recusal at ToolNode execution time.

Requires Python 3.10+ and is tested with exactly langgraph==1.2.12 and
langchain-core==1.6.5. Recusal itself remains Python 3.9+ and has no LangGraph dependency.

The hook is LangGraph-native: a user's existing @tool list goes unchanged into ToolNode and
recusal_gate(policy) is attached with wrap_tool_call. Every call in a model message crosses
the gate separately immediately before execution. Denials become error ToolMessages carrying
Recusal's reason; allow/defer execute normally. Async graphs attach recusal_agate(policy)
through awrap_tool_call and run with ainvoke, using the same decision logic.

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
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition
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


def _refusal(request, policy: Policy):
    """Return an error message for a refused call, or None to execute it."""
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
    return None


def recusal_gate(policy: Policy):
    """Return a synchronous ToolNode hook that gates each call before execution."""

    def gate(request, execute):
        refusal = _refusal(request, policy)
        if refusal is not None:
            return refusal
        return execute(request)

    return gate


def recusal_agate(policy: Policy):
    """Return an asynchronous ToolNode hook for awrap_tool_call."""

    async def gate(request, execute):
        refusal = _refusal(request, policy)
        if refusal is not None:
            return refusal
        return await execute(request)

    return gate


def tool_message(calls: list[dict[str, Any]]) -> AIMessage:
    """A scripted model message proposing tool calls; no model is invoked."""
    return AIMessage(content="", tool_calls=calls)


def _graph(node: ToolNode, calls: list[dict[str, Any]]):
    """Build the same scripted agent graph for sync and async invocation."""
    script = iter([tool_message(calls)])

    def scripted_agent(state: MessagesState) -> dict[str, Any]:
        return {"messages": [next(script, AIMessage(content="done"))]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", scripted_agent)
    builder.add_node("tools", node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile()


def run_in_graph(node: ToolNode, calls: list[dict[str, Any]]) -> list[ToolMessage]:
    """Run a ToolNode inside the MessagesState graph used by an agent."""
    messages = _graph(node, calls).invoke({"messages": []})["messages"]
    return [message for message in messages if isinstance(message, ToolMessage)]


async def run_in_graph_async(node: ToolNode, calls: list[dict[str, Any]]) -> list[ToolMessage]:
    """Run a ToolNode through ainvoke, exercising its async hook and async tools."""
    messages = (await _graph(node, calls).ainvoke({"messages": []}))["messages"]
    return [message for message in messages if isinstance(message, ToolMessage)]


def run_calls(
    calls: list[dict[str, Any]], *, policy: Policy = sql_scope_policy, gated: bool = True
) -> tuple[list[ToolMessage], list[tuple[str, dict[str, Any]]]]:
    EXECUTED.clear()
    tools = [echo, run_sql]
    node = ToolNode(tools, wrap_tool_call=recusal_gate(policy)) if gated else ToolNode(tools)
    return run_in_graph(node, calls), list(EXECUTED)


async def run_calls_async(
    calls: list[dict[str, Any]], *, policy: Policy = sql_scope_policy, gated: bool = True
) -> tuple[list[ToolMessage], list[tuple[str, dict[str, Any]]]]:
    EXECUTED.clear()
    tools = [echo, run_sql]
    node = ToolNode(tools, awrap_tool_call=recusal_agate(policy)) if gated else ToolNode(tools)
    return await run_in_graph_async(node, calls), list(EXECUTED)


def main() -> None:
    calls = [
        {"name": "echo", "args": {"text": "hello"}, "id": "1", "type": "tool_call"},
        {
            "name": "run_sql",
            "args": {"sql": "DELETE FROM audit_log"},
            "id": "2",
            "type": "tool_call",
        },
        {
            "name": "run_sql",
            "args": {"sql": "SELECT * FROM audit_log"},
            "id": "3",
            "type": "tool_call",
        },
    ]
    messages, executed = run_calls(calls)
    for message in messages:
        print(f"{message.name} [{message.status}]: {message.content}")
    print(f"tools that ran: {len(executed)}")


if __name__ == "__main__":
    main()
