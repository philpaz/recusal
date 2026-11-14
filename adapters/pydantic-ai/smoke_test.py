"""Prove the INSTALLED recusal-pydantic-ai wheel gates a real Pydantic AI agent.

Run it from a directory outside the repository, with an interpreter that has the built
wheels installed:

    python /path/to/adapters/pydantic-ai/smoke_test.py

It fails if the package it imports is the source tree and not an installed distribution:
a check run from the checkout would prove nothing about the wheel. No model, API key or
network is used.
"""

from __future__ import annotations

import os
import sys
import tempfile
from importlib import metadata
from pathlib import Path

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

# Python puts this script's directory first on the import path. Drop it, so nothing in
# the source tree can stand in for the installed package.
_HERE = Path(__file__).resolve().parent
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != _HERE]

import recusal_pydantic_ai  # noqa: E402
from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.messages import (  # noqa: E402
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel  # noqa: E402
from recusal_pydantic_ai import (  # noqa: E402
    SURFACE,
    RecusalAuditedGate,
    RecusalGate,
    is_gated,
)

import recusal  # noqa: E402
from recusal import Finding  # noqa: E402
from recusal.audit import AuditLog, verify  # noqa: E402

EXECUTED: list[str] = []


def run_sql(sql: str) -> str:
    """Record SQL text without connecting to a database."""
    EXECUTED.append(sql)
    return f"recorded: {sql}"


def policy(tool_name, tool_input):
    if str(tool_input.get("sql", "")).upper().startswith("DELETE"):
        return [Finding.fail("no_deletes", severity="CRITICAL", message="DELETE is not allowed")]
    return []


def scripted(calls):
    turns = {"n": 0}

    def model(messages, info):
        turns["n"] += 1
        if turns["n"] == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name="run_sql", args={"sql": sql}, tool_call_id=ident)
                    for ident, sql in calls
                ]
            )
        return ModelResponse(parts=[TextPart(content="done")])

    return FunctionModel(model)


def outcomes(result):
    return {
        part.tool_call_id: (part.outcome, part.content)
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }


def main() -> None:
    for module in (recusal, recusal_pydantic_ai):
        origin = Path(module.__file__).resolve()
        assert "site-packages" in origin.parts, f"{module.__name__} is not installed: {origin}"
    assert metadata.version("recusal-pydantic-ai") == recusal_pydantic_ai.__version__
    requires = metadata.requires("recusal-pydantic-ai") or []
    assert any(r.startswith("recusal") and "pydantic" not in r for r in requires), requires

    calls = [("c1", "SELECT 1"), ("c2", "DELETE FROM t")]

    # Negative control: without the gate the DELETE runs, so the refusal below is the gate's.
    EXECUTED.clear()
    ungated = Agent(scripted(calls), tools=[run_sql])
    assert not is_gated(ungated)
    ungated.run_sync("go")
    assert sorted(EXECUTED) == ["DELETE FROM t", "SELECT 1"], EXECUTED

    EXECUTED.clear()
    gated = Agent(scripted(calls), tools=[run_sql], capabilities=[RecusalGate(policy)])
    assert is_gated(gated)
    answers = outcomes(gated.run_sync("go"))
    assert EXECUTED == ["SELECT 1"], EXECUTED
    assert answers["c1"][0] == "success", answers
    assert answers["c2"][0] == "failed" and "DELETE is not allowed" in answers["c2"][1], answers

    EXECUTED.clear()
    with tempfile.TemporaryDirectory() as folder:
        log = AuditLog(path=os.path.join(folder, "audit.jsonl"))
        audited = Agent(
            scripted(calls), tools=[run_sql], capabilities=[RecusalAuditedGate(policy, log=log)]
        )
        audited.run_sync("go")
        assert EXECUTED == ["SELECT 1"], EXECUTED
        by_id = {entry["action"]["tool_use_id"]: entry["action"] for entry in log.entries}
        assert by_id["c2"]["decision"] == "deny" and by_id["c2"]["surface"] == SURFACE, by_id
        assert by_id["c1"]["decision"] != "deny", by_id
        intact, problems = verify(log.entries)
        assert intact, problems

    print(
        f"recusal-pydantic-ai {recusal_pydantic_ai.__version__} with recusal "
        f"{recusal.__version__}: installed wheel gates and records"
    )


if __name__ == "__main__":
    main()
