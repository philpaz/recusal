"""Prove the INSTALLED recusal-google-adk wheel gates a real Google ADK agent.

Run it from a directory outside the repository, with an interpreter that has the built
wheels installed:

    python /path/to/adapters/google-adk/smoke_test.py

It fails if the package it imports is the source tree and not an installed distribution:
a check run from the checkout would prove nothing about the wheel. No model, API key or
network is used.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any, AsyncGenerator

# Python puts this script's directory first on the import path. Drop it, so nothing in
# the source tree can stand in for the installed package.
_HERE = Path(__file__).resolve().parent
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != _HERE]

import recusal_google_adk  # noqa: E402
from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.models import BaseLlm, LlmRequest, LlmResponse  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.adk.tools import FunctionTool  # noqa: E402
from google.genai import types  # noqa: E402
from recusal_google_adk import (  # noqa: E402
    SURFACE,
    RecusalPlugin,
    audited_agent,
    audited_plugin,
    gated_agent,
    is_gated,
)

import recusal  # noqa: E402
from recusal import Finding  # noqa: E402
from recusal.audit import AuditLog, verify  # noqa: E402

EXECUTED: list[str] = []
CALLS = [("c1", "SELECT 1"), ("c2", "DELETE FROM t")]


def run_sql(sql: str) -> str:
    """Record SQL text without connecting to a database."""
    EXECUTED.append(sql)
    return f"recorded: {sql}"


def policy(tool_name, tool_input):
    if str(tool_input.get("sql", "")).upper().startswith("DELETE"):
        return [Finding.fail("no_deletes", severity="CRITICAL", message="DELETE is not allowed")]
    return []


class ScriptedLlm(BaseLlm):
    """Proposes the scripted calls on its first turn, then answers "done"."""

    turns: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.turns += 1
        if self.turns == 1:
            parts = [
                types.Part(
                    function_call=types.FunctionCall(id=ident, name="run_sql", args={"sql": sql})
                )
                for ident, sql in CALLS
            ]
        else:
            parts = [types.Part(text="done")]
        yield LlmResponse(content=types.Content(role="model", parts=parts))


def agent() -> LlmAgent:
    return LlmAgent(
        name="assistant", model=ScriptedLlm(model="scripted"), tools=[FunctionTool(run_sql)]
    )


async def _run(target: Any, plugins: Any) -> dict[str, Any]:
    runner = InMemoryRunner(agent=target, app_name="smoke", plugins=plugins)
    session = await runner.session_service.create_session(app_name="smoke", user_id="user")
    message = types.Content(role="user", parts=[types.Part(text="go")])
    responses: dict[str, Any] = {}
    async for event in runner.run_async(user_id="user", session_id=session.id, new_message=message):
        for part in event.content.parts if event.content and event.content.parts else []:
            if part.function_response is not None:
                responses[part.function_response.id] = part.function_response.response
    return responses


def run(target: Any, plugins: Any = None) -> dict[str, Any]:
    EXECUTED.clear()
    return asyncio.run(_run(target, plugins))


def main() -> None:
    for module in (recusal, recusal_google_adk):
        origin = Path(module.__file__).resolve()
        assert "site-packages" in origin.parts, f"{module.__name__} is not installed: {origin}"
    assert metadata.version("recusal-google-adk") == recusal_google_adk.__version__
    requires = metadata.requires("recusal-google-adk") or []
    assert any(r.startswith("recusal") and "google" not in r for r in requires), requires

    # Negative control: without the gate the DELETE runs, so the refusals below are the gate's.
    assert not is_gated(agent())
    run(agent())
    assert sorted(EXECUTED) == ["DELETE FROM t", "SELECT 1"], EXECUTED

    gated = gated_agent(agent(), policy)
    assert is_gated(gated)
    answers = run(gated)
    assert EXECUTED == ["SELECT 1"], EXECUTED
    assert "DELETE is not allowed" in answers["c2"]["error"], answers
    assert "error" not in answers["c1"], answers

    answers = run(agent(), plugins=[RecusalPlugin(policy)])
    assert EXECUTED == ["SELECT 1"], EXECUTED
    assert "DELETE is not allowed" in answers["c2"]["error"], answers

    for via_plugin in (False, True):
        with tempfile.TemporaryDirectory() as folder:
            log = AuditLog(path=os.path.join(folder, "audit.jsonl"))
            if via_plugin:
                run(agent(), plugins=[audited_plugin(policy, log)])
            else:
                run(audited_agent(agent(), policy, log))
            assert EXECUTED == ["SELECT 1"], EXECUTED
            by_id = {entry["action"]["tool_use_id"]: entry["action"] for entry in log.entries}
            assert by_id["c2"]["decision"] == "deny" and by_id["c2"]["surface"] == SURFACE, by_id
            assert by_id["c1"]["decision"] != "deny", by_id
            intact, problems = verify(log.entries)
            assert intact, problems

    print(
        f"recusal-google-adk {recusal_google_adk.__version__} with recusal "
        f"{recusal.__version__}: installed wheel gates and records"
    )


if __name__ == "__main__":
    main()
