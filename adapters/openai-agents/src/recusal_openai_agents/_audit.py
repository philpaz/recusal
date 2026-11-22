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
- a call refused before judgment (arguments that do not validate against the tool's
  parameters, a prepared value that is not JSON data): the arguments as proposed;
- a call whose arguments are not a JSON object, hold a non-finite number or are nested
  more than 64 levels deep: the argument text, or the JSON value it parsed to, under the
  key ``input``.

When an entry is written: when the guardrail runs, which is when the tool is about to
execute, after any approval, so a call a person rejected has no entry. Under the SDK's
``pre_approval_tool_input_guardrails=True`` the guardrail also runs before the approval:
a call then has an entry before a person decides, and a second one if it is approved.

What is not on the log: a call the SDK rejects before the guardrail runs (a tool the agent
does not have, a reused or a blank tool call id, arguments nested about a thousand levels
deep: each raises inside the SDK), and anything ``gated_agent`` refuses to gate at
construction, which raises before any agent runs.
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

    A blank tool call id is recorded as no id rather than refused. SDK 0.24.0 rejects a
    blank id itself, before the guardrail runs, so that case reaches this only on another
    seam or release."""

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
