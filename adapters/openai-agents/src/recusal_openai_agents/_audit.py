"""The audited gate: the same decision, with every call on a hash-chained ``AuditLog``.

``audited_agent(agent, policy, log)`` and ``audited_tools(tools, policy, log)`` are
``gated_agent`` and ``gate_tools`` with a ``recording`` that hands the plain gate's own
refusal function the arguments ``recusal.decide`` records with. There is no second
decision path. If the log cannot be written, or the recording raises, the call is refused,
so nothing runs unrecorded.

Each entry carries the surface ``openai_agents.tool_input_guardrail`` and the SDK's tool
call id, and a SHA-256 fingerprint of the arguments, never the arguments themselves.

What each entry is bound to:

- a call that is judged: the arguments the tool is called with, as the SDK prepares them
  from the tool's schema, defaults included;
- a call refused before judgment (arguments that are not a JSON object, a schema that
  transforms its input, a preparation the gate cannot read): the arguments as proposed.

What is not on the log: a call the SDK rejects before the guardrail runs (a tool the agent
does not have raises ``ModelBehaviorError``), and anything ``gated_agent`` refuses to gate
at construction, which raises before any agent runs.
"""

from __future__ import annotations

from typing import Any, Optional

from agents import Agent, FunctionTool

from recusal.audit import AuditLog

from ._gate import Policy, Recording, gate_tools, gated_agent

#: The surface label every entry carries: the SDK hook that adjudicated.
SURFACE = "openai_agents.tool_input_guardrail"


def recording(
    log: AuditLog, *, actor: Optional[str] = None, control: Optional[dict[str, Any]] = None
) -> Recording:
    """The recording arguments for each call, built in one place.

    A blank tool call id is recorded as no id rather than refused."""

    def for_call(ctx: Any) -> dict[str, Any]:
        ident = getattr(ctx, "tool_call_id", None)
        return {
            "audit": log,
            "surface": SURFACE,
            "tool_use_id": ident if isinstance(ident, str) and ident else None,
            "actor": actor,
            "control": control,
        }

    return for_call


def audited_tools(
    tools: list[Any],
    policy: Policy,
    log: AuditLog,
    *,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
) -> list[FunctionTool]:
    """``gate_tools`` with every decision of every guardrail on ``log``."""
    return gate_tools(tools, policy, recording(log, actor=actor, control=control))


def audited_agent(
    agent: Agent,
    policy: Policy,
    log: AuditLog,
    *,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
) -> Agent:
    """``gated_agent`` with every decision on ``log``."""
    return gated_agent(agent, policy, recording(log, actor=actor, control=control))
