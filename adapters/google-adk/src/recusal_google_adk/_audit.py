"""The audited gate: the same decision, with every call on a hash-chained ``AuditLog``.

``audited_agent(agent, policy, log)`` and ``audited_plugin(policy, log)`` are
``gated_agent`` and ``RecusalPlugin`` with a ``recording`` that hands the plain gate's own
refusal function the arguments ``recusal.decide`` records with. There is no second
decision path. If the log cannot be written, or the recording raises, the call is refused,
so nothing runs unrecorded.

Each entry carries the surface ``google_adk.before_tool_callback`` and the ADK's function
call id, and a SHA-256 fingerprint of the arguments, never the arguments themselves.

What each entry is bound to:

- a call that is judged: the prepared invocation the tool is called with, defaults
  completed and values validated;
- a call refused before judgment (a tool that is not a stock ``FunctionTool``, a signature
  the gate cannot judge, arguments that do not validate, a plugin that is not the last
  before-tool seam): the arguments as proposed.

What is not on the log: a call another before-tool callback answered before the gate ran
(the tool did not run either), and anything ``gated_agent`` refuses to gate at
construction, which raises before any agent exists.
"""

from __future__ import annotations

from typing import Any, Optional

from google.adk.agents import BaseAgent

from recusal.audit import AuditLog

from ._gate import Policy, Recording, RecusalPlugin, gated_agent

#: The surface label every entry carries: the ADK hook that adjudicated.
SURFACE = "google_adk.before_tool_callback"


def recording(
    log: AuditLog, *, actor: Optional[str] = None, control: Optional[dict[str, Any]] = None
) -> Recording:
    """The recording arguments for each call, built in one place for both forms of the gate.

    The ADK gives every function call an id (it generates one when the model sends none);
    a blank id is recorded as no id rather than refused."""

    def for_call(tool_context: Any) -> dict[str, Any]:
        ident = getattr(tool_context, "function_call_id", None)
        return {
            "audit": log,
            "surface": SURFACE,
            "tool_use_id": ident if isinstance(ident, str) and ident else None,
            "actor": actor,
            "control": control,
        }

    return for_call


def audited_agent(
    agent: BaseAgent,
    policy: Policy,
    log: AuditLog,
    *,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
) -> BaseAgent:
    """``gated_agent`` with every decision, of the agent and of every agent under it, on
    ``log``."""
    return gated_agent(agent, policy, recording(log, actor=actor, control=control))


def audited_plugin(
    policy: Policy,
    log: AuditLog,
    *,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
) -> RecusalPlugin:
    """``RecusalPlugin`` with every decision on ``log``; list it last, as with the plain one."""
    return RecusalPlugin(policy, recording=recording(log, actor=actor, control=control))
