"""Prove the INSTALLED recusal-crewai wheel gates a real CrewAI crew.

Run it from a directory outside the repository, with an interpreter that has the built
wheels installed:

    python /path/to/adapters/crewai/smoke_test.py

It fails if the package it imports is the source tree and not an installed distribution:
a check run from the checkout would prove nothing about the wheel. A scripted model
proposes the calls; no provider, API key or network is used.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# offline, and quiet: no telemetry, no tracing banner. Set before CrewAI is imported.
for _var, _value in (
    ("CREWAI_DISABLE_TELEMETRY", "true"),
    ("CREWAI_TRACING_ENABLED", "false"),
    ("OTEL_SDK_DISABLED", "true"),
):
    os.environ.setdefault(_var, _value)

# Python puts this script's directory first on the import path. Drop it, so nothing in
# the source tree can stand in for the installed package.
_HERE = Path(__file__).resolve().parent
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != _HERE]

import recusal_crewai  # noqa: E402
from crewai import Agent, Crew, Task  # noqa: E402
from crewai.llms.base_llm import BaseLLM  # noqa: E402
from crewai.tools import tool  # noqa: E402
from recusal_crewai import SURFACE, install_audited_hooks, install_recusal_hooks  # noqa: E402

import recusal  # noqa: E402
from recusal import Finding  # noqa: E402
from recusal.audit import AuditLog, verify  # noqa: E402

EXECUTED: list[str] = []
CALLS = [("c1", "SELECT 1"), ("c2", "DELETE FROM t")]


@tool("run_sql")
def run_sql(sql: str) -> str:
    """Record SQL text without connecting to a database."""
    EXECUTED.append(sql)
    return f"recorded: {sql}"


def policy(tool_name, tool_input):
    if str(tool_input.get("sql", "")).upper().startswith("DELETE"):
        return [Finding.fail("no_deletes", severity="CRITICAL", message="DELETE is not allowed")]
    return []


class ScriptedLLM(BaseLLM):
    """Proposes the scripted calls on its first turn, then answers "done"."""

    def __init__(self) -> None:
        super().__init__(model="scripted")
        self.turns = 0

    def call(self, messages, tools=None, callbacks=None, available_functions=None, **_: Any):
        self.turns += 1
        if self.turns == 1:
            return [
                SimpleNamespace(
                    id=ident,
                    type="function",
                    function=SimpleNamespace(name="run_sql", arguments=json.dumps({"sql": sql})),
                )
                for ident, sql in CALLS
            ]
        return "Final Answer: done"

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 100_000


def run() -> dict[str, str]:
    """Run one task through a Crew; return what the model received for each call, by id."""
    EXECUTED.clear()
    agent = Agent(
        role="assistant",
        goal="do the task",
        backstory="an offline agent",
        llm=ScriptedLLM(),
        tools=[run_sql],
        verbose=False,
    )
    task = Task(description="go", expected_output="done", agent=agent)
    Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
    executor = agent.agent_executor
    return {
        str(message.get("tool_call_id")): str(message.get("content"))
        for message in (getattr(executor.state, "messages", []) if executor is not None else [])
        if isinstance(message, dict) and message.get("role") == "tool"
    }


def main() -> None:
    for module in (recusal, recusal_crewai):
        origin = Path(module.__file__).resolve()
        assert "site-packages" in origin.parts, f"{module.__name__} is not installed: {origin}"
    assert metadata.version("recusal-crewai") == recusal_crewai.__version__
    requires = metadata.requires("recusal-crewai") or []
    assert any(r.startswith("recusal") and "crewai" not in r for r in requires), requires

    # Negative control: without the hooks the DELETE runs, so the refusals below are the gate's.
    run()
    assert sorted(EXECUTED) == ["DELETE FROM t", "SELECT 1"], EXECUTED

    uninstall = install_recusal_hooks(policy)
    try:
        answers = run()
    finally:
        uninstall()
    assert EXECUTED == ["SELECT 1"], EXECUTED
    assert "recorded: SELECT 1" in answers["c1"], answers
    assert "DELETE is not allowed" in answers["c2"], answers

    # the hooks are gone once uninstalled: the registry is process-global, so this matters
    run()
    assert sorted(EXECUTED) == ["DELETE FROM t", "SELECT 1"], EXECUTED

    with tempfile.TemporaryDirectory() as folder:
        log = AuditLog(path=os.path.join(folder, "audit.jsonl"))
        uninstall = install_audited_hooks(policy, log)
        try:
            run()
        finally:
            uninstall()
        assert EXECUTED == ["SELECT 1"], EXECUTED
        # no call id on a CrewAI entry, and one turn's entries land in any order
        decisions = sorted(entry["action"]["decision"] for entry in log.entries)
        assert len(decisions) == 2 and decisions.count("deny") == 1, decisions
        assert {entry["action"]["surface"] for entry in log.entries} == {SURFACE}
        assert {entry["action"]["control"]["crewai_agent"] for entry in log.entries} == {
            "assistant"
        }
        intact, problems = verify(log.entries)
        assert intact, problems

    print(
        f"recusal-crewai {recusal_crewai.__version__} with recusal "
        f"{recusal.__version__}: installed wheel gates and records"
    )


if __name__ == "__main__":
    main()
