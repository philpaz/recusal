"""
Offline Pydantic AI example: gate every tool call with a Recusal capability.

The gate is the ``recusal-pydantic-ai`` package (``adapters/pydantic-ai`` in this
repository): ``RecusalGate``, ``recusal_before_tool_execute`` and ``is_gated`` are
imported from it. This file is the offline demonstration around it: a scripted model,
two tools that only record, and the SQL scope policy.

Requires Python 3.11+ (Pydantic AI's minimum) and is tested with exactly
pydantic-ai-slim==2.64.0. Recusal itself remains Python 3.9+ and has no Pydantic AI
dependency.

The hook is Pydantic AI's own. An ``Agent`` runs each capability's ``before_tool_execute``
hook after a tool call's arguments are validated and immediately before the tool runs, with
the call, the tool definition and the validated argument dict the tool will receive. The
hook returns the arguments to run with, or raises ``ToolFailed`` to answer the call with a
failed tool result instead of running it. ``RecusalGate(policy)`` judges the call with
``recusal.decide`` and, on a refusal, raises ``ToolFailed(reason)``, so the
model reads Recusal's reason on its next turn, the tool never runs, and no retry budget is
spent (a refusal is not a model mistake to correct). Every tool call in a model turn is
judged separately. Every toolset the agent runs, ``tools=``, ``toolsets=`` or a capability's
own tools, executes through the same ``ToolManager``, so the hook sees each call by name and
validated arguments; output tools (structured output) are not tool calls and skip it.

Supported arguments are data-only: JSON scalars, lists and string-keyed dicts. Containers
are detached by construction. Models, dataclasses, Enum members and other objects are
refused before policy evaluation. Declare data-only parameters (a ``str`` or ``int`` in
place of an Enum) and resolve application types inside the tool body. No application copy
method or serializer is called.

Two ways to attach it:

- ``Agent(..., capabilities=[RecusalGate(policy)])`` for every run of the agent, or
  ``agent.run(..., capabilities=[RecusalGate(policy)])`` for one run. The gate declares
  itself ``innermost``, so Pydantic AI sorts it after every ordinary capability however the
  list is written: ``before_tool_execute`` hooks run in capability order and each one hands
  its arguments to the next, so a hook that ran after the gate could change the arguments
  the gate judged. Only another ``innermost`` capability listed after the gate still runs
  after it; list the gate last.
- ``Hooks(before_tool_execute=recusal_before_tool_execute(policy))`` is the same gate as a
  plain hook function. ``Hooks`` declares no ordering, so this form must be the last
  capability in the list; the tests show the bypass when it is not.

What the hook cannot see, so the example names it rather than skips it:

- an agent another agent calls from inside a tool function: its tool calls run under its
  own capabilities, not the caller's. Give every agent its own gate, at construction or
  per run inside the delegating tool; ``is_gated(agent)`` reads only the agent it is given.
- a tool call whose arguments fail validation: Pydantic AI answers it with a retry prompt
  before any execute hook runs, and the tool does not run either.

No model, API key or network is used: ``scripted_model`` proposes the calls from a script,
and run_sql only records SQL text.

Run after installing:
    pip install pydantic-ai-slim==2.64.0
    python examples/pydantic_ai_gate.py
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

PYDANTIC_AI_VERSION = "2.64.0"

if sys.version_info < (3, 11):
    print(
        "pydantic_ai_gate.py requires Python 3.11+ "
        f"(pydantic-ai-slim {PYDANTIC_AI_VERSION}); "
        f"this is Python {sys.version_info[0]}.{sys.version_info[1]}.",
        file=sys.stderr,
    )
    raise SystemExit(2)

# Pydantic AI prints an observability banner to stderr on the first run of a process;
# the example's output is the gate's, so the banner is off here.
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

try:
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import Hooks
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel
except ImportError as exc:
    raise SystemExit(
        f"Install the example dependency first: pydantic-ai-slim=={PYDANTIC_AI_VERSION}"
    ) from exc

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
# In a checkout the adapter is read from the tree; installed, `pip install
# recusal-pydantic-ai` provides the same package and this path does not exist.
sys.path.insert(0, os.path.join(_ROOT, "adapters", "pydantic-ai", "src"))

from recusal_pydantic_ai import (  # noqa: E402
    Policy,
    RecusalGate,
    is_gated,
    recusal_before_tool_execute,
)

from examples.sql_scope_policy import policy as sql_scope_policy  # noqa: E402

__all__ = ["Policy", "RecusalGate", "is_gated", "recusal_before_tool_execute"]

EXECUTED: list[tuple[str, dict[str, Any]]] = []


def echo(text: str) -> str:
    """Return harmless text."""
    EXECUTED.append(("echo", {"text": text}))
    return text


def run_sql(sql: str) -> str:
    """Record SQL text without connecting to a database."""
    EXECUTED.append(("run_sql", {"sql": sql}))
    return f"recorded SQL: {sql}"


# --- offline fixtures: a scripted model, no network --------------------------------------


def scripted_model(calls: list[tuple[str, Any, str]]) -> FunctionModel:
    """A model that proposes the scripted calls on its first turn, then answers "done"."""
    turns = {"n": 0}

    def model(messages: list, info: AgentInfo) -> ModelResponse:
        turns["n"] += 1
        if turns["n"] == 1 and calls:
            parts = [
                ToolCallPart(tool_name=name, args=args, tool_call_id=call_id)
                for name, args, call_id in calls
            ]
            return ModelResponse(parts=parts)
        return ModelResponse(parts=[TextPart(content="done")])

    return FunctionModel(model)


def scripted_agent(
    calls: list[tuple[str, Any, str]],
    tools: Optional[list] = None,
    capabilities: Optional[list] = None,
) -> Agent:
    return Agent(
        scripted_model(calls),
        tools=[echo, run_sql] if tools is None else tools,
        capabilities=capabilities,
    )


def responses_of(result: Any) -> dict[str, tuple[str, Any]]:
    """Each tool call's answer by id: ``(outcome, content)`` from the run's messages."""
    out: dict[str, tuple[str, Any]] = {}
    for message in result.all_messages():
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                out[part.tool_call_id] = (part.outcome, part.content)
    return out


def run_calls(
    calls: list[tuple[str, Any, str]],
    *,
    policy: Policy = sql_scope_policy,
    gated: bool = True,
    via_hooks: bool = False,
) -> tuple[dict[str, tuple[str, Any]], list[tuple[str, dict[str, Any]]]]:
    """Propose the calls to an agent with echo and run_sql; return answers and what ran."""
    EXECUTED.clear()
    capabilities: Optional[list] = None
    if gated and via_hooks:
        capabilities = [Hooks(before_tool_execute=recusal_before_tool_execute(policy))]
    elif gated:
        capabilities = [RecusalGate(policy)]
    result = scripted_agent(calls, capabilities=capabilities).run_sync("go")
    return responses_of(result), list(EXECUTED)


def main() -> None:
    calls = [
        ("echo", {"text": "hello"}, "c1"),
        ("run_sql", {"sql": "DELETE FROM audit_log"}, "c2"),
        ("run_sql", {"sql": "SELECT * FROM audit_log"}, "c3"),
    ]
    for label, via_hooks in (("RecusalGate", False), ("Hooks(before_tool_execute)", True)):
        responses, executed = run_calls(calls, via_hooks=via_hooks)
        print(f"{label}:")
        for name, _, call_id in calls:
            outcome, content = responses[call_id]
            print(f"  {name} ({call_id}): {outcome}: {content}")
        print(f"  tools that ran: {len(executed)}")


if __name__ == "__main__":
    main()
