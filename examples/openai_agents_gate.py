"""
Offline OpenAI Agents SDK example: gate function tools with a Recusal tool input guardrail.

Requires Python 3.10+ and is tested with exactly openai-agents==0.24.0 and openai==3.34.0.
Recusal itself remains Python 3.9+ and has no OpenAI dependency.

The hook is the SDK's own: a tool input guardrail runs immediately before a function tool
executes, and recusal_guardrail(policy) judges the call with recusal.claude_code.decide. A
refusal becomes the call's output, so the model sees Recusal's reason on its next turn and
the tool never runs. Every call in a model turn is judged separately, and Runner.run,
Runner.run_streamed and a human approval (needs_approval) all pass through it: the guardrail
runs when the tool is about to execute, after any approval.

What the guardrail cannot see, so gated_agent refuses it rather than skipping it:

- every tool that is not a FunctionTool: hosted tools that run on OpenAI's servers (web
  search, file search, code interpreter, image generation, hosted MCP, tool search) and
  local ones the SDK executes itself (shell, local shell, apply patch, computer, custom,
  programmatic tool calling). The rule is default-deny: anything but a FunctionTool is
  refused, so a tool type added in a later SDK release is refused too;
- MCP servers attached with mcp_servers=, and handoffs to other agents;
- a sub-agent behind agent.as_tool(...) whose own tools are not gated: the guardrail sees
  only the input handed to the sub-agent, never the calls it then makes. Gate the sub-agent
  first, gated_agent(sub, policy).as_tool(...), and the parent accepts it;
- the experimental Codex tool, which runs its own commands;
- Agent subclasses such as SandboxAgent, which add tools of their own at run time.

gated_agent checks the agent when you call it. Build the agent completely, then gate it;
a tool appended to the gated agent's list afterwards is not gated.

No model, API key or database is used: agents.testing.ScriptedModel proposes the calls,
and run_sql only records SQL text.

Run after installing:
    pip install openai-agents==0.24.0 openai==3.34.0
    python examples/openai_agents_gate.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sys
from typing import Any, Callable, List

OPENAI_AGENTS_VERSION = "0.24.0"
OPENAI_VERSION = "3.34.0"

if sys.version_info < (3, 10):
    print(
        "openai_agents_gate.py requires Python 3.10+ "
        f"(openai-agents {OPENAI_AGENTS_VERSION}); "
        f"this is Python {sys.version_info[0]}.{sys.version_info[1]}.",
        file=sys.stderr,
    )
    raise SystemExit(2)

try:
    from agents import (
        Agent,
        FunctionTool,
        RunConfig,
        Runner,
        ToolGuardrailFunctionOutput,
        function_tool,
        tool_input_guardrail,
    )
    from agents.testing import ScriptedModel, assistant_message, function_call
except ImportError as exc:
    raise SystemExit(
        "Install the example dependencies first: "
        f"openai-agents=={OPENAI_AGENTS_VERSION} openai=={OPENAI_VERSION}"
    ) from exc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.sql_scope_policy import policy as sql_scope_policy  # noqa: E402
from recusal.claude_code import decide  # noqa: E402

Policy = Callable[[str, dict], List[Any]]
EXECUTED: list[tuple[str, dict[str, Any]]] = []
GUARDRAIL_NAME = "recusal"


@function_tool
def echo(text: str) -> str:
    """Return harmless text."""
    EXECUTED.append(("echo", {"text": text}))
    return text


@function_tool
def run_sql(sql: str) -> str:
    """Record SQL text without connecting to a database."""
    EXECUTED.append(("run_sql", {"sql": sql}))
    return f"recorded SQL: {sql}"


def _refusal(tool_name: str, arguments: str, policy: Policy) -> str | None:
    """Return the reason a call is refused, or None to run it.

    The SDK decodes a function tool's arguments with json.loads, so the guardrail decodes
    the same string the same way and judges exactly what the tool will receive.
    """
    try:
        args = json.loads(arguments or "{}")
    except ValueError:
        return f"Recusal refused `{tool_name}`: arguments are not valid JSON"
    if not isinstance(args, dict):
        return f"Recusal refused `{tool_name}`: arguments are not a JSON object"
    try:
        decision, reason = decide(tool_name, args, policy)
    except Exception as exc:
        return f"Recusal failed closed (policy error): {exc}"
    return reason if decision == "deny" else None


def recusal_guardrail(policy: Policy):
    """Return a tool input guardrail that judges each function tool call before it runs."""

    def gate(data) -> ToolGuardrailFunctionOutput:
        reason = _refusal(data.context.tool_name, data.context.tool_arguments, policy)
        if reason is not None:
            return ToolGuardrailFunctionOutput.reject_content(reason)
        return ToolGuardrailFunctionOutput.allow()

    gate.recusal_gate = True  # type: ignore[attr-defined]
    return tool_input_guardrail(gate, name=GUARDRAIL_NAME)


def _carries_gate(tool: FunctionTool) -> bool:
    return any(
        getattr(guardrail.guardrail_function, "recusal_gate", False)
        for guardrail in tool.tool_input_guardrails or ()
    )


def _ungateable_reason(agent: Any) -> str | None:
    """Why an agent can run a tool the guardrail never sees, or None if it cannot."""
    if type(agent) is not Agent:
        return (
            f"{type(agent).__name__} is not a plain Agent; a subclass can add tools at run "
            "time that the guardrail never sees"
        )
    if agent.mcp_servers:
        return "it has mcp_servers, whose tools the guardrail cannot see"
    if agent.handoffs:
        return "it has handoffs, which hand the conversation to another agent's tools"
    return None


def is_gated(agent: Any, _seen: set[int] | None = None) -> bool:
    """True if every tool this agent can run passes through a Recusal guardrail."""
    seen = set() if _seen is None else _seen
    if id(agent) in seen:
        return True
    seen.add(id(agent))
    if _ungateable_reason(agent) is not None:
        return False
    for tool in agent.tools:
        if not isinstance(tool, FunctionTool) or not _carries_gate(tool):
            return False
        if getattr(tool, "_is_codex_tool", False):
            return False
        if getattr(tool, "_is_agent_tool", False):
            if not is_gated(getattr(tool, "_agent_instance", None), seen):
                return False
    return True


def gate_tools(tools: list[Any], policy: Policy) -> list[FunctionTool]:
    """Return copies of the tools with the Recusal guardrail added; the originals are unchanged.

    Raises TypeError, naming the tool, for anything the guardrail cannot see.
    """
    guardrail = recusal_guardrail(policy)
    gated = []
    for tool in tools:
        name = getattr(tool, "name", None) or "(unnamed)"
        if not isinstance(tool, FunctionTool):
            raise TypeError(
                f"Recusal cannot gate {type(tool).__name__} {name!r}: tool input guardrails "
                "run only on function tools"
            )
        if getattr(tool, "_is_codex_tool", False):
            raise TypeError(
                f"Recusal cannot gate the Codex tool {name!r}: it runs its own commands"
            )
        if getattr(tool, "_is_agent_tool", False) and not is_gated(
            getattr(tool, "_agent_instance", None)
        ):
            raise TypeError(
                f"Recusal cannot gate agent tool {name!r}: its agent's own tool calls are not "
                "gated. Use gated_agent(sub_agent, policy).as_tool(...)"
            )
        gated.append(
            dataclasses.replace(
                tool, tool_input_guardrails=[*(tool.tool_input_guardrails or ()), guardrail]
            )
        )
    return gated


def gated_agent(agent: Agent, policy: Policy) -> Agent:
    """Return a copy of the agent whose every tool call is judged by Recusal.

    Raises ValueError if the agent can run anything the guardrail cannot see.
    """
    reason = _ungateable_reason(agent)
    if reason is not None:
        raise ValueError(f"Recusal cannot gate agent {agent.name!r}: {reason}")
    return agent.clone(tools=gate_tools(list(agent.tools), policy))


def scripted_model(calls: list[tuple[str, Any, str]]) -> ScriptedModel:
    """A model that proposes the calls in one turn, then answers; nothing is contacted."""
    turn = [function_call(name, arguments, call_id=call_id) for name, arguments, call_id in calls]
    return ScriptedModel([turn, [assistant_message("done")]])


def tool_outputs(model: ScriptedModel) -> dict[str, str]:
    """The output of each call, keyed by call_id, as the model received it on its next turn."""
    call = model.last_call
    assert call is not None and not isinstance(call.input, str)
    return {
        item["call_id"]: item["output"]
        for item in call.input
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    }


async def run_agent(agent: Agent, *, streamed: bool = False) -> None:
    """Run the agent offline through Runner.run or Runner.run_streamed."""
    config = RunConfig(tracing_disabled=True)
    if streamed:
        result = Runner.run_streamed(agent, "go", run_config=config)
        async for _ in result.stream_events():
            pass
    else:
        await Runner.run(agent, "go", run_config=config)


def run_calls(
    calls: list[tuple[str, Any, str]],
    *,
    policy: Policy = sql_scope_policy,
    gated: bool = True,
    streamed: bool = False,
) -> tuple[dict[str, str], list[tuple[str, dict[str, Any]]]]:
    """Propose the calls to an agent with echo and run_sql; return outputs and what ran."""
    EXECUTED.clear()
    model = scripted_model(calls)
    agent = Agent(name="assistant", model=model, tools=[echo, run_sql])
    if gated:
        agent = gated_agent(agent, policy)
    asyncio.run(run_agent(agent, streamed=streamed))
    return tool_outputs(model), list(EXECUTED)


def main() -> None:
    calls = [
        ("echo", {"text": "hello"}, "c1"),
        ("run_sql", {"sql": "DELETE FROM audit_log"}, "c2"),
        ("run_sql", {"sql": "SELECT * FROM audit_log"}, "c3"),
    ]
    outputs, executed = run_calls(calls)
    for name, _, call_id in calls:
        print(f"{name} ({call_id}): {outputs[call_id]}")
    print(f"tools that ran: {len(executed)}")


if __name__ == "__main__":
    main()
