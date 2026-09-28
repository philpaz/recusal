"""
Offline LangGraph example: gate every proposed tool call with Recusal before it runs.

Requires Python 3.10+ and is tested with exactly langgraph==1.2.12 and
langchain-core==1.6.5. The Recusal package itself still supports Python 3.9+
and does not depend on LangGraph.

Hook choice: each harmless tool function is wrapped at the call boundary. That keeps
Recusal immediately before the effect, preserves LangGraph's normal graph routing, and
lets a denial become the tool result that the scripted agent sees. A custom ToolNode
would couple this example to more LangGraph internals, while LangGraph interrupt/approval
would add a separate human-approval mechanism instead of demonstrating Recusal's
deterministic policy gate.

No LLM or API key is used. The SQL tool below only records the text it was asked to
run; it never connects to a database.

Run after installing:
    pip install langgraph==1.2.12 langchain-core==1.6.5
    python examples/langgraph_gate.py
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.sql_scope_policy import policy as sql_scope_policy  # noqa: E402
from recusal.claude_code import decide  # noqa: E402

LANGGRAPH_VERSION = "1.2.12"
LANGCHAIN_CORE_VERSION = "1.6.5"

Policy = Callable[[str, dict], List[Any]]
ToolFunction = Callable[..., str]
ToolCall = Dict[str, Any]
ExecutedCall = Tuple[str, Dict[str, Any]]


class GraphState(TypedDict):
    calls: List[ToolCall]
    index: int
    current: Optional[ToolCall]
    results: List[str]


def _require_python(version_info: Any = None) -> None:
    version = sys.version_info if version_info is None else version_info
    major, minor = int(version[0]), int(version[1])
    if (major, minor) < (3, 10):
        print(
            "langgraph_gate.py requires Python 3.10+ "
            f"(LangGraph {LANGGRAPH_VERSION}); this is Python {major}.{minor}.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _langgraph():
    _require_python()
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise SystemExit(
            "Install the example dependencies first: "
            f"langgraph=={LANGGRAPH_VERSION} langchain-core=={LANGCHAIN_CORE_VERSION}"
        ) from exc
    return StateGraph, START, END


def gate_tool(tool_name: str, function: ToolFunction, policy: Policy) -> ToolFunction:
    def wrapped(**tool_input: Any) -> str:
        decision, reason = decide(tool_name, tool_input, policy)
        if decision == "deny":
            return f"REFUSED: {reason}"
        return function(**tool_input)

    return wrapped


def run_scripted(
    calls: List[ToolCall],
    *,
    policy: Policy = sql_scope_policy,
    gated: bool = True,
    executed: Optional[List[ExecutedCall]] = None,
) -> Tuple[GraphState, List[ExecutedCall]]:
    StateGraph, START, END = _langgraph()
    execution_log = [] if executed is None else executed

    def echo(text: str) -> str:
        execution_log.append(("echo", {"text": text}))
        return f"echo: {text}"

    def run_sql(sql: str) -> str:
        execution_log.append(("run_sql", {"sql": sql}))
        return f"recorded SQL: {sql}"

    raw_tools: Dict[str, ToolFunction] = {"echo": echo, "run_sql": run_sql}
    tools = (
        {name: gate_tool(name, function, policy) for name, function in raw_tools.items()}
        if gated
        else raw_tools
    )

    def scripted_agent(state: GraphState) -> Dict[str, Any]:
        index = state["index"]
        current = state["calls"][index] if index < len(state["calls"]) else None
        return {"current": current}

    def route_after_agent(state: GraphState) -> str:
        return "tools" if state["current"] is not None else "done"

    def tool_node(state: GraphState) -> Dict[str, Any]:
        current = state["current"]
        if current is None:
            raise RuntimeError("tool node reached without a proposed call")
        name = current["name"]
        arguments = dict(current.get("args", {}))
        if name not in tools:
            raise KeyError(f"unknown scripted tool: {name}")
        result = tools[name](**arguments)
        return {
            "index": state["index"] + 1,
            "current": None,
            "results": [*state["results"], result],
        }

    builder = StateGraph(GraphState)
    builder.add_node("agent", scripted_agent)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "done": END})
    builder.add_edge("tools", "agent")
    graph = builder.compile()

    state = graph.invoke({"calls": calls, "index": 0, "current": None, "results": []})
    return state, execution_log


def main(version_info: Any = None) -> None:
    _require_python(version_info)
    calls = [
        {"name": "echo", "args": {"text": "hello"}},
        {"name": "run_sql", "args": {"sql": "DELETE FROM audit_log"}},
        {"name": "run_sql", "args": {"sql": "SELECT * FROM audit_log"}},
    ]
    state, executed = run_scripted(calls)
    for call, result in zip(calls, state["results"]):
        print(f"{call['name']}: {result}")
    print(f"executed tool calls: {len(executed)}")


if __name__ == "__main__":
    main()
