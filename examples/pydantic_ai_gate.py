"""
Offline Pydantic AI example: gate every tool call with a Recusal capability.

Requires Python 3.11+ (Pydantic AI's minimum) and is tested with exactly
pydantic-ai-slim==2.64.0. Recusal itself remains Python 3.9+ and has no Pydantic AI
dependency.

The hook is Pydantic AI's own. An ``Agent`` runs each capability's ``before_tool_execute``
hook after a tool call's arguments are validated and immediately before the tool runs, with
the call, the tool definition and the validated argument dict the tool will receive. The
hook returns the arguments to run with, or raises ``ToolFailed`` to answer the call with a
failed tool result instead of running it. ``RecusalGate(policy)`` judges the call with
``recusal.claude_code.decide`` and, on a refusal, raises ``ToolFailed(reason)``, so the
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

import math
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional

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
    from pydantic_ai.capabilities import (
        AbstractCapability,
        CapabilityOrdering,
        CombinedCapability,
        Hooks,
    )
    from pydantic_ai.exceptions import ToolFailed
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from pydantic_ai.tools import ToolDefinition
except ImportError as exc:
    raise SystemExit(
        f"Install the example dependency first: pydantic-ai-slim=={PYDANTIC_AI_VERSION}"
    ) from exc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.sql_scope_policy import policy as sql_scope_policy  # noqa: E402
from recusal.claude_code import decide  # noqa: E402

Policy = Callable[[str, dict], List[Any]]
EXECUTED: list[tuple[str, dict[str, Any]]] = []


def echo(text: str) -> str:
    """Return harmless text."""
    EXECUTED.append(("echo", {"text": text}))
    return text


def run_sql(sql: str) -> str:
    """Record SQL text without connecting to a database."""
    EXECUTED.append(("run_sql", {"sql": sql}))
    return f"recorded SQL: {sql}"


def _detached(value: Any, ancestors: tuple[int, ...] = ()) -> Any:
    """Detach the supported data domain without invoking application copying/serialization.

    Exact container/scalar types avoid subclasses with custom iteration or copy behavior.
    Every Enum member is refused, whatever its value: a member is one shared object, and
    attributes its class sets on it (an ``__init__`` that stores a dict, say) would be the
    same object at policy and at execution. Passing only ``.value`` would hide that state
    from the policy while the tool still used it, so that is no repair either.
    """
    if type(value) in (str, int, bool, type(None)):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError("non-finite numbers are not JSON data")
        return value
    if isinstance(value, Enum):
        raise TypeError(
            "Enum arguments are unsupported; declare data-only parameters "
            "and resolve application types inside the tool body"
        )
    if type(value) not in (dict, list):
        raise TypeError("arguments must be data-only")
    if id(value) in ancestors:
        raise TypeError("cyclic arguments are not JSON data")
    ancestors = (*ancestors, id(value))
    if type(value) is list:
        return [_detached(item, ancestors) for item in value]
    if any(type(key) is not str for key in value):
        raise TypeError("argument keys must be strings")
    return {key: _detached(item, ancestors) for key, item in value.items()}


def _refusal(tool_name: str, args: Any, policy: Policy) -> Optional[str]:
    """Return the reason a call is refused, or None to run it.

    Pydantic AI hands the hook the validated argument dict the tool will be called with, so
    the policy judges exactly what would run. A non-dict is refused: the tool would not
    accept it either, and the policy cannot read it.

    The policy judges detached data, built by ``_detached``, and the tool runs with the
    validated dict itself. Unsupported values refuse before the policy is called. No
    model projection, application copy method or serializer participates in the decision.
    """
    if not isinstance(args, dict):
        return f"Recusal refused `{tool_name}`: arguments are not a mapping"
    try:
        observed = _detached(args)
    except Exception as exc:  # noqa: BLE001, an unsupported observation must not run
        return (
            f"Recusal refused `{tool_name}`: arguments cannot be detached for judging "
            f"({type(exc).__name__}); declare data-only parameters and validate inside the tool body"
        )
    try:
        decision, reason = decide(tool_name, observed, policy, safe_errors=True)
    except Exception as exc:  # noqa: BLE001, a buggy policy must not silently disable the gate
        return f"Recusal failed closed (policy error): RECUSAL_POLICY_ERROR ({type(exc).__name__})"
    return reason if decision == "deny" else None


def recusal_before_tool_execute(policy: Policy):
    """Return a ``before_tool_execute`` hook that judges each tool call before it runs.

    For ``Hooks(before_tool_execute=...)``. A refusal is raised as ``ToolFailed(reason)``,
    which Pydantic AI turns into a failed tool result the model reads on its next turn;
    the returned arguments run the tool unchanged.
    """

    async def gate(ctx: Any, *, call: ToolCallPart, tool_def: ToolDefinition, args: dict) -> dict:
        reason = _refusal(call.tool_name, args, policy)
        if reason is not None:
            raise ToolFailed(reason)
        return args

    gate.recusal_gate = True  # type: ignore[attr-defined]
    return gate


@dataclass
class RecusalGate(AbstractCapability[Any]):
    """The gate as a capability: ``Agent(..., capabilities=[RecusalGate(policy)])``.

    Declares the ``innermost`` position, so it runs after every ordinary capability's
    ``before_tool_execute`` and judges the arguments as they reach the tool.
    """

    policy: Policy = field(default=sql_scope_policy)

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")

    async def before_tool_execute(
        self, ctx: Any, *, call: ToolCallPart, tool_def: ToolDefinition, args: dict
    ) -> dict:
        reason = _refusal(call.tool_name, args, self.policy)
        if reason is not None:
            raise ToolFailed(reason)
        return args


def _leaves(capability: Any) -> list:
    if isinstance(capability, CombinedCapability):
        return [leaf for child in capability.capabilities for leaf in _leaves(child)]
    return [capability]


def is_gated(agent: Agent) -> bool:
    """True if a ``RecusalGate`` is among the agent's own capabilities.

    Reads only this agent: an agent it calls from inside a tool function runs under its own
    capabilities and must be checked, and gated, itself.
    """
    return any(isinstance(leaf, RecusalGate) for leaf in _leaves(agent.root_capability))


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
