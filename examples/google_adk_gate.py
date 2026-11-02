"""
Offline Google ADK example: gate every tool call with a Recusal before-tool callback.

Requires Python 3.11+ (the ADK's minimum) and is tested with exactly google-adk==2.13.0 and
google-genai==2.34.0. Recusal itself remains Python 3.9+ and has no ADK dependency.

The hook is the ADK's own. An ``LlmAgent`` runs its ``before_tool_callback`` immediately
before each tool call, with the tool, the arguments the tool will receive and the tool
context; a callback that returns a dict answers the call instead of running the tool.
``recusal_before_tool(policy)`` judges the call with ``recusal.claude_code.decide`` and, on
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
- ``RecusalPlugin(policy)`` is the same gate as a runner plugin: plugin callbacks run before
  any agent's callbacks, for every agent the runner drives, so one plugin covers an app whose
  agents you do not build yourself. List it last among the plugins for the same reason.

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
from typing import Any, AsyncGenerator, Callable, List, Optional

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
    from google.adk.agents.loop_agent import LoopAgent
    from google.adk.agents.parallel_agent import ParallelAgent
    from google.adk.agents.sequential_agent import SequentialAgent
    from google.adk.models import BaseLlm, LlmRequest, LlmResponse
    from google.adk.plugins import BasePlugin
    from google.adk.runners import InMemoryRunner
    from google.adk.tools import AgentTool, BaseTool, FunctionTool, ToolContext
    from google.genai import types
except ImportError as exc:
    raise SystemExit(
        "Install the example dependencies first: "
        f"google-adk=={GOOGLE_ADK_VERSION} google-genai=={GOOGLE_GENAI_VERSION}"
    ) from exc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.sql_scope_policy import policy as sql_scope_policy  # noqa: E402
from recusal.claude_code import decide  # noqa: E402

Policy = Callable[[str, dict], List[Any]]
EXECUTED: list[tuple[str, dict[str, Any]]] = []

#: Workflow agents run no tools of their own; their ``sub_agents`` do. ``gated_agent``
#: gates those and leaves the workflow agent's own class alone.
_WORKFLOW_AGENTS = (SequentialAgent, ParallelAgent, LoopAgent)


def echo(text: str) -> str:
    """Return harmless text."""
    EXECUTED.append(("echo", {"text": text}))
    return text


def run_sql(sql: str) -> str:
    """Record SQL text without connecting to a database."""
    EXECUTED.append(("run_sql", {"sql": sql}))
    return f"recorded SQL: {sql}"


def _refusal(tool_name: str, args: Any, policy: Policy) -> Optional[str]:
    """Return the reason a call is refused, or None to run it.

    The ADK hands the callback the same ``args`` dict the tool will be called with, so the
    policy judges exactly what would run. A non-dict is refused: the tool would not accept
    it either, and the policy cannot read it.
    """
    if not isinstance(args, dict):
        return f"Recusal refused `{tool_name}`: arguments are not a mapping"
    try:
        decision, reason = decide(tool_name, args, policy)
    except Exception as exc:  # noqa: BLE001, a buggy policy must not silently disable the gate
        return f"Recusal failed closed (policy error): {exc}"
    return reason if decision == "deny" else None


def recusal_before_tool(policy: Policy):
    """Return a ``before_tool_callback`` that judges each tool call before it runs.

    A refusal is returned as the tool's response, ``{"error": reason}``, so the model sees
    why on its next turn; ``None`` lets the ADK run the tool.
    """

    def gate(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext):
        reason = _refusal(tool.name, args, policy)
        if reason is not None:
            return {"error": reason}
        return None

    gate.recusal_gate = True  # type: ignore[attr-defined]
    return gate


class RecusalPlugin(BasePlugin):
    """The same gate as a runner plugin: every agent the runner drives, before any agent
    callback. ``InMemoryRunner(agent, plugins=[..., RecusalPlugin(policy)])``."""

    def __init__(self, policy: Policy, name: str = "recusal"):
        super().__init__(name=name)
        self._policy = policy

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> Optional[dict[str, Any]]:
        reason = _refusal(tool.name, tool_args, self._policy)
        return {"error": reason} if reason is not None else None


def _callbacks(agent: LlmAgent) -> list:
    existing = agent.before_tool_callback
    if existing is None:
        return []
    return list(existing) if isinstance(existing, list) else [existing]


def _carries_gate(agent: LlmAgent) -> bool:
    return any(getattr(cb, "recusal_gate", False) for cb in _callbacks(agent))


def _ungateable_reason(agent: Any) -> Optional[str]:
    """Why an agent can run something the callback never sees, or None if it cannot."""
    if isinstance(agent, _WORKFLOW_AGENTS):
        return None
    if type(agent) is not LlmAgent:
        return (
            f"{type(agent).__name__} is not a plain LlmAgent; a subclass can run tools or "
            "code the before-tool callback never sees"
        )
    if agent.code_executor is not None:
        return "it has a code_executor; model-written code runs outside the tool path"
    return None


def is_gated(agent: Any, _seen: Optional[set] = None) -> bool:
    """True if every tool call this agent, its sub-agents and its agent tools can make
    passes through a Recusal before-tool callback."""
    seen = set() if _seen is None else _seen
    if id(agent) in seen:
        return True
    seen.add(id(agent))
    if _ungateable_reason(agent) is not None:
        return False
    if isinstance(agent, LlmAgent):
        if not _carries_gate(agent):
            return False
        for tool in agent.tools:
            if isinstance(tool, AgentTool) and not is_gated(tool.agent, seen):
                return False
    return all(is_gated(sub, seen) for sub in agent.sub_agents)


def gated_agent(agent: BaseAgent, policy: Policy) -> BaseAgent:
    """Return a copy of the agent whose every tool call is judged by Recusal.

    The gate is appended as the LAST before-tool callback; ``sub_agents`` and the agent
    behind every ``AgentTool`` are gated the same way. Raises ValueError if the agent, or
    any agent under it, can run something the callback cannot see.
    """
    reason = _ungateable_reason(agent)
    if reason is not None:
        raise ValueError(f"Recusal cannot gate agent {agent.name!r}: {reason}")
    update: dict[str, Any] = {"sub_agents": [gated_agent(sub, policy) for sub in agent.sub_agents]}
    if isinstance(agent, LlmAgent):
        tools = []
        for tool in agent.tools:
            if isinstance(tool, AgentTool):
                tools.append(
                    AgentTool(
                        gated_agent(tool.agent, policy), skip_summarization=tool.skip_summarization
                    )
                )
            else:
                tools.append(tool)
        update["tools"] = tools
        update["before_tool_callback"] = _callbacks(agent) + [recusal_before_tool(policy)]
    return agent.clone(update=update)


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
        tools=[FunctionTool(echo), FunctionTool(run_sql)] if tools is None else tools,
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
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    """Propose the calls to an agent with echo and run_sql; return responses and what ran."""
    EXECUTED.clear()
    agent: BaseAgent = scripted_agent(calls)
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
