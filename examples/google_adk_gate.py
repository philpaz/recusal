"""
Offline Google ADK example: gate every tool call with a Recusal before-tool callback.

The gate is the ``recusal-google-adk`` package (``adapters/google-adk`` in this
repository): ``gated_agent``, ``RecusalPlugin``, ``recusal_before_tool`` and ``is_gated``
are imported from it. This file is the offline demonstration around it: a scripted
model, three tools that only record, and the SQL scope policy.

Requires Python 3.11+ (the ADK's minimum) and is tested with exactly google-adk==2.13.0 and
google-genai==2.34.0. Recusal itself remains Python 3.9+ and has no ADK dependency.

The hook is the ADK's own. An ``LlmAgent`` runs its ``before_tool_callback`` immediately
before each tool call, with the tool, the arguments the tool will receive and the tool
context; a callback that returns a dict answers the call instead of running the tool.
``recusal_before_tool(policy)`` judges the call with ``recusal.decide`` and, on
a refusal, returns ``{"error": reason}`` so the model reads Recusal's reason on its next turn
and the tool never runs. Every call in a model turn is judged separately. The ADK routes
every ``BaseTool`` through the same path (function tools, long-running tools, agent tools,
MCP toolsets), so the callback sees each of them by name and arguments.

Two ways to attach it:

- ``gated_agent(agent, policy)`` returns a copy of the agent with the gate appended as the
  LAST before-tool callback, and gates its ``sub_agents`` and the agent behind every
  ``AgentTool`` the same way. Last, because callbacks run in order and a callback listed
  after the gate could change the arguments the gate judged; a callback listed before it
  that answers the call itself means the tool did not run.
- ``RecusalPlugin(policy)`` is the same gate as a runner plugin, for every agent the runner
  drives, so one plugin covers an app whose agents you do not build yourself. Plugin
  callbacks run before any agent's callbacks, so the plugin cannot be placed after an
  agent's own ``before_tool_callback``, which could change the arguments the plugin judged.
  Its contract is therefore explicit: it judges a call only when nothing it cannot see runs
  after it, that is, when the executing agent carries no before-tool callback other than a
  Recusal gate and no plugin listed after it overrides ``before_tool_callback``; otherwise it
  refuses the call, naming the callback or plugin, and the fix (``gated_agent`` for the
  agent, which appends the gate last among its callbacks; listing the plugin last). A call
  whose invocation context does not expose the executing agent is refused too.

What the callback cannot see, so ``gated_agent`` refuses rather than skips it:

- a ``code_executor`` on the agent: model-written code runs outside the tool path;
- a ``sub_agent`` or an ``AgentTool`` whose agent is not itself gated: its tool calls run
  under its own callbacks, not the parent's (``gated_agent`` gates them recursively, and
  refuses an agent class it does not know how to gate);
- an ``LlmAgent`` subclass: it can add behavior the gate was not written for.

No model, API key or network is used: ``ScriptedLlm`` proposes the calls from a script,
and run_sql only records SQL text.

Run after installing:
    pip install google-adk==2.13.0 google-genai==2.34.0
    python examples/google_adk_gate.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, AsyncGenerator, Optional

GOOGLE_ADK_VERSION = "2.13.0"
GOOGLE_GENAI_VERSION = "2.34.0"

if sys.version_info < (3, 11):
    print(
        "google_adk_gate.py requires Python 3.11+ "
        f"(google-adk {GOOGLE_ADK_VERSION}); "
        f"this is Python {sys.version_info[0]}.{sys.version_info[1]}.",
        file=sys.stderr,
    )
    raise SystemExit(2)

try:
    from google.adk.agents import BaseAgent, LlmAgent
    from google.adk.models import BaseLlm, LlmRequest, LlmResponse
    from google.adk.runners import InMemoryRunner
    from google.adk.tools import FunctionTool
    from google.genai import types
except ImportError as exc:
    raise SystemExit(
        "Install the example dependencies first: "
        f"google-adk=={GOOGLE_ADK_VERSION} google-genai=={GOOGLE_GENAI_VERSION}"
    ) from exc

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
# In a checkout the adapter is read from the tree; installed, `pip install
# recusal-google-adk` provides the same package and this path does not exist.
sys.path.insert(0, os.path.join(_ROOT, "adapters", "google-adk", "src"))

from recusal_google_adk import (  # noqa: E402
    Policy,
    Recording,
    RecusalPlugin,
    gated_agent,
    is_gated,
    recusal_before_tool,
)

from examples.sql_scope_policy import policy as sql_scope_policy  # noqa: E402

__all__ = [
    "Policy",
    "Recording",
    "RecusalPlugin",
    "gated_agent",
    "is_gated",
    "recusal_before_tool",
]

EXECUTED: list[tuple[str, dict[str, Any]]] = []


def echo(text: str) -> str:
    """Return harmless text."""
    EXECUTED.append(("echo", {"text": text}))
    return text


def run_sql(sql: str) -> str:
    """Record SQL text without connecting to a database."""
    EXECUTED.append(("run_sql", {"sql": sql}))
    return f"recorded SQL: {sql}"


def charge(amount: int = 100) -> str:
    """Record a hypothetical charge with a default the model may omit; contacts nothing."""
    EXECUTED.append(("charge", {"amount": amount}))
    return "recorded"


# --- offline fixtures: a scripted model, no network --------------------------------------


class ScriptedLlm(BaseLlm):
    """A model that proposes the scripted calls on its first turn, then answers "done"."""

    calls: list[tuple[str, Any, str]] = []
    turns: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.turns += 1
        if self.turns == 1 and self.calls:
            parts = [
                types.Part(function_call=types.FunctionCall(id=call_id, name=name, args=args))
                for name, args, call_id in self.calls
            ]
        else:
            parts = [types.Part(text="done")]
        yield LlmResponse(content=types.Content(role="model", parts=parts))


def scripted_agent(calls: list[tuple[str, Any, str]], tools: Optional[list] = None) -> LlmAgent:
    return LlmAgent(
        name="assistant",
        model=ScriptedLlm(model="scripted", calls=list(calls)),
        tools=[FunctionTool(echo), FunctionTool(run_sql), FunctionTool(charge)]
        if tools is None
        else tools,
    )


async def run_agent(agent: BaseAgent, plugins: Optional[list] = None) -> dict[str, Any]:
    """Run the agent offline through InMemoryRunner; return each call's response by id."""
    runner = InMemoryRunner(agent=agent, app_name="recusal-adk-example", plugins=plugins)
    session = await runner.session_service.create_session(
        app_name="recusal-adk-example", user_id="user"
    )
    message = types.Content(role="user", parts=[types.Part(text="go")])
    responses: dict[str, Any] = {}
    async for event in runner.run_async(user_id="user", session_id=session.id, new_message=message):
        for part in event.content.parts if event.content and event.content.parts else []:
            if part.function_response is not None:
                responses[part.function_response.id] = part.function_response.response
    return responses


def run_calls(
    calls: list[tuple[str, Any, str]],
    *,
    policy: Policy = sql_scope_policy,
    gated: bool = True,
    via_plugin: bool = False,
    tools: Optional[list] = None,
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    """Propose the calls to an agent with echo, run_sql and charge; return responses and
    what ran."""
    EXECUTED.clear()
    agent: BaseAgent = scripted_agent(calls, tools=tools)
    plugins = None
    if gated and via_plugin:
        plugins = [RecusalPlugin(policy)]
    elif gated:
        agent = gated_agent(agent, policy)
    responses = asyncio.run(run_agent(agent, plugins=plugins))
    return responses, list(EXECUTED)


def main() -> None:
    calls = [
        ("echo", {"text": "hello"}, "c1"),
        ("run_sql", {"sql": "DELETE FROM audit_log"}, "c2"),
        ("run_sql", {"sql": "SELECT * FROM audit_log"}, "c3"),
    ]
    for label, via_plugin in (("before_tool_callback", False), ("RecusalPlugin", True)):
        responses, executed = run_calls(calls, via_plugin=via_plugin)
        print(f"{label}:")
        for name, _, call_id in calls:
            print(f"  {name} ({call_id}): {responses[call_id]}")
        print(f"  tools that ran: {len(executed)}")


if __name__ == "__main__":
    main()
