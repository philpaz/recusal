"""Prove the INSTALLED recusal-langgraph wheel gates a real LangGraph ToolNode.

Run it from a directory outside the repository, with an interpreter that has the built
wheels installed:

    python /path/to/adapters/langgraph/smoke_test.py

It fails if the package it imports is the source tree and not an installed distribution:
a check run from the checkout would prove nothing about the wheel. A scripted graph
proposes the calls; no model, API key or network is used.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

# Python puts this script's directory first on the import path. Drop it, so nothing in
# the source tree can stand in for the installed package.
_HERE = Path(__file__).resolve().parent
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != _HERE]

import recusal_langgraph  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.graph import START, MessagesState, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode, tools_condition  # noqa: E402
from recusal_langgraph import (  # noqa: E402
    SURFACE,
    recusal_agate,
    recusal_audited_gate,
    recusal_gate,
)

import recusal  # noqa: E402
from recusal import Finding  # noqa: E402
from recusal.audit import AuditLog, verify  # noqa: E402

EXECUTED: list[str] = []
CALLS = [
    {"name": "run_sql", "args": {"sql": "SELECT 1"}, "id": "c1", "type": "tool_call"},
    {"name": "run_sql", "args": {"sql": "DELETE FROM t"}, "id": "c2", "type": "tool_call"},
]


@tool
def run_sql(sql: str) -> str:
    """Record SQL text without connecting to a database."""
    EXECUTED.append(sql)
    return f"recorded: {sql}"


def policy(tool_name, tool_input):
    if str(tool_input.get("sql", "")).upper().startswith("DELETE"):
        return [Finding.fail("no_deletes", severity="CRITICAL", message="DELETE is not allowed")]
    return []


def graph(node: ToolNode):
    script = iter([AIMessage(content="", tool_calls=CALLS)])

    def scripted_agent(state: MessagesState) -> dict[str, Any]:
        return {"messages": [next(script, AIMessage(content="done"))]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", scripted_agent)
    builder.add_node("tools", node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile()


def answers(messages: list) -> dict[str, tuple[str, str]]:
    return {
        m.tool_call_id: (m.status, str(m.content)) for m in messages if isinstance(m, ToolMessage)
    }


def run(node: ToolNode) -> dict[str, tuple[str, str]]:
    EXECUTED.clear()
    return answers(graph(node).invoke({"messages": []})["messages"])


async def arun(node: ToolNode) -> dict[str, tuple[str, str]]:
    EXECUTED.clear()
    return answers((await graph(node).ainvoke({"messages": []}))["messages"])


def main() -> None:
    for module in (recusal, recusal_langgraph):
        origin = Path(module.__file__).resolve()
        assert "site-packages" in origin.parts, f"{module.__name__} is not installed: {origin}"
    assert metadata.version("recusal-langgraph") == recusal_langgraph.__version__
    requires = metadata.requires("recusal-langgraph") or []
    assert any(r.startswith("recusal") and "lang" not in r for r in requires), requires

    # Negative control: without the gate the DELETE runs, so the refusals below are the gate's.
    run(ToolNode([run_sql]))
    assert sorted(EXECUTED) == ["DELETE FROM t", "SELECT 1"], EXECUTED

    got = run(ToolNode([run_sql], wrap_tool_call=recusal_gate(policy)))
    assert EXECUTED == ["SELECT 1"], EXECUTED
    assert got["c1"] == ("success", "recorded: SELECT 1"), got
    assert got["c2"][0] == "error" and "DELETE is not allowed" in got["c2"][1], got

    got = asyncio.run(arun(ToolNode([run_sql], awrap_tool_call=recusal_agate(policy))))
    assert EXECUTED == ["SELECT 1"], EXECUTED
    assert got["c2"][0] == "error" and "DELETE is not allowed" in got["c2"][1], got

    with tempfile.TemporaryDirectory() as folder:
        log = AuditLog(path=os.path.join(folder, "audit.jsonl"))
        run(ToolNode([run_sql], wrap_tool_call=recusal_audited_gate(policy, log)))
        assert EXECUTED == ["SELECT 1"], EXECUTED
        by_id = {entry["action"]["tool_use_id"]: entry["action"] for entry in log.entries}
        assert by_id["c2"]["decision"] == "deny" and by_id["c2"]["surface"] == SURFACE, by_id
        assert by_id["c1"]["decision"] != "deny", by_id
        intact, problems = verify(log.entries)
        assert intact, problems

    print(
        f"recusal-langgraph {recusal_langgraph.__version__} with recusal "
        f"{recusal.__version__}: installed wheel gates and records"
    )


if __name__ == "__main__":
    main()
