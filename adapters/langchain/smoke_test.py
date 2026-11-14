"""Prove the INSTALLED recusal-langchain wheel gates a real LangChain create_agent agent.

Run it from a directory outside the repository, with an interpreter that has the built
wheels installed (recusal, recusal-langgraph and recusal-langchain):

    python /path/to/adapters/langchain/smoke_test.py

It fails if a package it imports is the source tree and not an installed distribution:
a check run from the checkout would prove nothing about the wheels. A scripted model
proposes the calls; no provider, API key or network is used.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from importlib import metadata
from pathlib import Path

# Python puts this script's directory first on the import path. Drop it, so nothing in
# the source tree can stand in for the installed package.
_HERE = Path(__file__).resolve().parent
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != _HERE]

import recusal_langchain  # noqa: E402
import recusal_langgraph  # noqa: E402
from langchain.agents import create_agent  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from recusal_langchain import (  # noqa: E402
    SURFACE,
    agate_middleware,
    audited_gate_middleware,
    gate_middleware,
)

import recusal  # noqa: E402
from recusal import Finding  # noqa: E402
from recusal.audit import AuditLog, verify  # noqa: E402

EXECUTED: list[str] = []
CALLS = [
    {"name": "run_sql", "args": {"sql": "SELECT 1"}, "id": "c1", "type": "tool_call"},
    {"name": "run_sql", "args": {"sql": "DELETE FROM t"}, "id": "c2", "type": "tool_call"},
]
INPUT = {"messages": [HumanMessage(content="go")]}


@tool
def run_sql(sql: str) -> str:
    """Record SQL text without connecting to a database."""
    EXECUTED.append(sql)
    return f"recorded: {sql}"


def policy(tool_name, tool_input):
    if str(tool_input.get("sql", "")).upper().startswith("DELETE"):
        return [Finding.fail("no_deletes", severity="CRITICAL", message="DELETE is not allowed")]
    return []


class ScriptedChatModel(GenericFakeChatModel):
    """Offline response stream with a no-op tool binder; no provider is called."""

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self


def agent(middleware: list):
    model = ScriptedChatModel(
        messages=iter([AIMessage(content="", tool_calls=CALLS), AIMessage(content="done")])
    )
    return create_agent(model=model, tools=[run_sql], middleware=middleware)


def answers(result) -> dict[str, tuple[str, str]]:
    return {
        m.tool_call_id: (m.status, str(m.content))
        for m in result["messages"]
        if isinstance(m, ToolMessage)
    }


def main() -> None:
    for module in (recusal, recusal_langgraph, recusal_langchain):
        origin = Path(module.__file__).resolve()
        assert "site-packages" in origin.parts, f"{module.__name__} is not installed: {origin}"
    assert metadata.version("recusal-langchain") == recusal_langchain.__version__
    requires = metadata.requires("recusal-langchain") or []
    assert any(r.startswith("recusal-langgraph") for r in requires), requires
    assert any(r.startswith("recusal") and "lang" not in r for r in requires), requires

    # Negative control: without the gate the DELETE runs, so the refusals below are the gate's.
    EXECUTED.clear()
    agent([]).invoke(INPUT)
    assert sorted(EXECUTED) == ["DELETE FROM t", "SELECT 1"], EXECUTED

    EXECUTED.clear()
    got = answers(agent([gate_middleware(policy)]).invoke(INPUT))
    assert EXECUTED == ["SELECT 1"], EXECUTED
    assert got["c1"] == ("success", "recorded: SELECT 1"), got
    assert got["c2"][0] == "error" and "DELETE is not allowed" in got["c2"][1], got

    EXECUTED.clear()
    got = answers(asyncio.run(agent([agate_middleware(policy)]).ainvoke(INPUT)))
    assert EXECUTED == ["SELECT 1"], EXECUTED
    assert got["c2"][0] == "error" and "DELETE is not allowed" in got["c2"][1], got

    EXECUTED.clear()
    with tempfile.TemporaryDirectory() as folder:
        log = AuditLog(path=os.path.join(folder, "audit.jsonl"))
        agent([audited_gate_middleware(policy, log)]).invoke(INPUT)
        assert EXECUTED == ["SELECT 1"], EXECUTED
        by_id = {entry["action"]["tool_use_id"]: entry["action"] for entry in log.entries}
        assert by_id["c2"]["decision"] == "deny" and by_id["c2"]["surface"] == SURFACE, by_id
        assert by_id["c1"]["decision"] != "deny", by_id
        intact, problems = verify(log.entries)
        assert intact, problems

    print(
        f"recusal-langchain {recusal_langchain.__version__} with recusal-langgraph "
        f"{recusal_langgraph.__version__} and recusal {recusal.__version__}: "
        "installed wheels gate and record"
    )


if __name__ == "__main__":
    main()
