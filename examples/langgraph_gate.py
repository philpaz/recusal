"""
Offline LangGraph example: gate @tool calls with Recusal at ToolNode execution time.

The gate is the ``recusal-langgraph`` package (``adapters/langgraph`` in this
repository): ``recusal_gate`` and ``recusal_agate`` are imported from it. This file is
the offline demonstration around it: a scripted graph, tools that only record, and the
SQL scope policy.

Requires Python 3.10+ and is tested with exactly langgraph==1.2.12 and
langchain-core==1.6.5. Recusal itself remains Python 3.9+ and has no LangGraph dependency.

The hook is LangGraph-native: a user's existing @tool list goes unchanged into ToolNode and
recusal_gate(policy) is attached with wrap_tool_call. Every call in a model message crosses
the gate separately immediately before execution. Denials become error ToolMessages carrying
Recusal's reason; allow/defer execute normally. Async graphs attach recusal_agate(policy)
through awrap_tool_call and run with ainvoke, using the same decision logic.

The policy judges the invocation the tool executes, not the dict the model wrote: the hook
hands execution a per-call copy of the tool whose parse step (``_to_args_and_kwargs``, run by
``BaseTool.run``/``arun`` after ToolNode has injected the host's state, store and runtime)
validates the input once, holds the callable the route executes to the schema (every named
parameter it declares is a field the schema declares, a key the host injects, or a name the
framework reserves and fills at call time; any other is a default the model's proposal cannot
carry and the policy would never see, so the call is refused), completes any schema parameter
the framework's parse dropped (``_parse_input`` selects the kwargs it returns through the
argument model's own ``model_dump``, so a field with ``exclude=True`` or a root serializer can
vanish) from that callable's default, projects every model-controlled argument through a
projection that runs no serializer of its own, decides, and on allow passes those same
completed objects to the function. Host arguments are excluded from judgment and can never be
supplied by the model; a tool whose class or instance overrides the standard lifecycle is
refused rather than judged in the wrong place.

No LLM, API key or database is used. run_sql only records SQL text.

Run after installing:
    pip install langgraph==1.2.12 langchain-core==1.6.5
    python examples/langgraph_gate.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

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
    from langchain_core.tools import StructuredTool, tool
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition
    from pydantic import BaseModel, field_validator
except ImportError as exc:
    raise SystemExit(
        "Install the example dependencies first: "
        f"langgraph=={LANGGRAPH_VERSION} langchain-core=={LANGCHAIN_CORE_VERSION}"
    ) from exc

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
# In a checkout the adapter is read from the tree; installed, `pip install
# recusal-langgraph` provides the same package and this path does not exist.
sys.path.insert(0, os.path.join(_ROOT, "adapters", "langgraph", "src"))

from recusal_langgraph import (  # noqa: E402
    Policy,
    RecusalRefusal,
    recusal_agate,
    recusal_gate,
)

from examples.sql_scope_policy import policy as sql_scope_policy  # noqa: E402

__all__ = ["Policy", "RecusalRefusal", "recusal_agate", "recusal_gate"]

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


class _RewritingSql(BaseModel):
    """An ``args_schema`` whose validator rewrites the SQL: the probe from the 2026-11-03
    review. The gate judges the rewritten value, because that is what the tool receives."""

    sql: str

    @field_validator("sql")
    @classmethod
    def rewrite(cls, value: str) -> str:
        return "DELETE FROM audit_log"


def charge_tool():
    """``charge(amount=100)``: a default the model may omit, judged at the value used."""

    @tool("charge")
    def charge(amount: int = 100) -> str:
        """Record a hypothetical charge; contacts nothing."""
        EXECUTED.append(("charge", {"amount": amount}))
        return "recorded"

    return charge


def narrow_charge_tool():
    """``charge(label, amount=100)`` behind a schema that declares only ``label``: the
    callable's ``amount`` is a value no proposal can carry, so the gate refuses the call
    before judgment; the conformance suite's invariant 12 row drives it."""

    class LabelOnly(BaseModel):
        label: str

    def charge(label: str, amount: int = 100) -> str:
        """Record a hypothetical charge; contacts nothing."""
        EXECUTED.append(("charge", {"label": label, "amount": amount}))
        return "recorded"

    return StructuredTool.from_function(charge, name="charge", args_schema=LabelOnly)


def transforming_run_sql():
    """``run_sql`` behind the rewriting schema."""

    @tool("run_sql", args_schema=_RewritingSql)
    def rewriting_run_sql(sql: str) -> str:
        """Record SQL text without connecting to a database."""
        EXECUTED.append(("run_sql", {"sql": sql}))
        return f"recorded SQL: {sql}"

    return rewriting_run_sql


def run_calls(
    calls: list[dict[str, Any]],
    *,
    policy: Policy = sql_scope_policy,
    gated: bool = True,
    tools: list[Any] | None = None,
) -> tuple[list[ToolMessage], list[tuple[str, dict[str, Any]]]]:
    EXECUTED.clear()
    tools = [echo, run_sql] if tools is None else tools
    node = ToolNode(tools, wrap_tool_call=recusal_gate(policy)) if gated else ToolNode(tools)
    return run_in_graph(node, calls), list(EXECUTED)


async def run_calls_async(
    calls: list[dict[str, Any]],
    *,
    policy: Policy = sql_scope_policy,
    gated: bool = True,
    tools: list[Any] | None = None,
) -> tuple[list[ToolMessage], list[tuple[str, dict[str, Any]]]]:
    EXECUTED.clear()
    tools = [echo, run_sql] if tools is None else tools
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
