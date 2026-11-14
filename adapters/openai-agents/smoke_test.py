"""Prove the INSTALLED recusal-openai-agents wheel gates a real OpenAI Agents SDK agent.

Run it from a directory outside the repository, with an interpreter that has the built
wheels installed:

    python /path/to/adapters/openai-agents/smoke_test.py

It fails if the package it imports is the source tree and not an installed distribution:
a check run from the checkout would prove nothing about the wheel. The SDK's own scripted
model proposes the calls; no API key or network is used.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from importlib import metadata
from pathlib import Path

# Python puts this script's directory first on the import path. Drop it, so nothing in
# the source tree can stand in for the installed package.
_HERE = Path(__file__).resolve().parent
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != _HERE]

import recusal_openai_agents  # noqa: E402
from agents import Agent, RunConfig, Runner, function_tool  # noqa: E402
from agents.testing import ScriptedModel, assistant_message, function_call  # noqa: E402
from recusal_openai_agents import SURFACE, audited_agent, gated_agent, is_gated  # noqa: E402

import recusal  # noqa: E402
from recusal import Finding  # noqa: E402
from recusal.audit import AuditLog, verify  # noqa: E402

EXECUTED: list[str] = []
CALLS = [("c1", "SELECT 1"), ("c2", "DELETE FROM t")]


@function_tool
def run_sql(sql: str) -> str:
    """Record SQL text without connecting to a database."""
    EXECUTED.append(sql)
    return f"recorded: {sql}"


def policy(tool_name, tool_input):
    if str(tool_input.get("sql", "")).upper().startswith("DELETE"):
        return [Finding.fail("no_deletes", severity="CRITICAL", message="DELETE is not allowed")]
    return []


def run(gate) -> dict[str, str]:
    """Run a fresh agent, passed through ``gate``; return each call's output by id."""
    EXECUTED.clear()
    turn = [
        function_call("run_sql", json.dumps({"sql": sql}), call_id=ident) for ident, sql in CALLS
    ]
    model = ScriptedModel([turn, [assistant_message("done")]])
    agent = gate(Agent(name="assistant", model=model, tools=[run_sql]))
    asyncio.run(Runner.run(agent, "go", run_config=RunConfig(tracing_disabled=True)))
    call = model.last_call
    assert call is not None and not isinstance(call.input, str)
    return {
        item["call_id"]: item["output"]
        for item in call.input
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    }


def main() -> None:
    for module in (recusal, recusal_openai_agents):
        origin = Path(module.__file__).resolve()
        assert "site-packages" in origin.parts, f"{module.__name__} is not installed: {origin}"
    assert metadata.version("recusal-openai-agents") == recusal_openai_agents.__version__
    requires = metadata.requires("recusal-openai-agents") or []
    assert any(r.startswith("recusal") and "openai" not in r for r in requires), requires

    # Negative control: without the gate the DELETE runs, so the refusals below are the gate's.
    run(lambda agent: agent)
    assert sorted(EXECUTED) == ["DELETE FROM t", "SELECT 1"], EXECUTED

    assert not is_gated(Agent(name="plain", tools=[run_sql]))
    assert is_gated(gated_agent(Agent(name="gated", tools=[run_sql]), policy))
    outputs = run(lambda agent: gated_agent(agent, policy))
    assert EXECUTED == ["SELECT 1"], EXECUTED
    assert outputs["c1"] == "recorded: SELECT 1", outputs
    assert "DELETE is not allowed" in outputs["c2"], outputs

    with tempfile.TemporaryDirectory() as folder:
        log = AuditLog(path=os.path.join(folder, "audit.jsonl"))
        run(lambda agent: audited_agent(agent, policy, log))
        assert EXECUTED == ["SELECT 1"], EXECUTED
        by_id = {entry["action"]["tool_use_id"]: entry["action"] for entry in log.entries}
        assert by_id["c2"]["decision"] == "deny" and by_id["c2"]["surface"] == SURFACE, by_id
        assert by_id["c1"]["decision"] != "deny", by_id
        intact, problems = verify(log.entries)
        assert intact, problems

    print(
        f"recusal-openai-agents {recusal_openai_agents.__version__} with recusal "
        f"{recusal.__version__}: installed wheel gates and records"
    )


if __name__ == "__main__":
    main()
