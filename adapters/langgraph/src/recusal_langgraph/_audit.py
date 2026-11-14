"""The audited gate: the same decision, with every call on a hash-chained ``AuditLog``.

``recusal_audited_gate(policy, log)`` and ``recusal_audited_agate(policy, log)`` call the
plain gate's own judging function with the recording arguments passed through to
``recusal.decide``, which writes the entry before returning. There is no second decision
path. If the log cannot be written the decision becomes a deny, so nothing runs unrecorded.

Each entry carries the surface ``langgraph.toolnode`` and LangGraph's tool call id, and a
SHA-256 fingerprint of the arguments, never the arguments themselves.

What each entry is bound to:

- a call that is judged: the invocation the tool executes, validated and completed, with
  host-injected arguments left out;
- a call refused before judgment (a tool that overrides the standard lifecycle, input that
  does not validate, a parameter the schema omits): the arguments as proposed.

What is not on the log: a call another ``wrap_tool_call`` hook answered without reaching
the gate.
"""

from __future__ import annotations

from typing import Any, Optional

from recusal.audit import AuditLog

from ._gate import Policy, _judge

#: The surface label every entry carries: the runtime that adjudicated, LangGraph's ToolNode.
SURFACE = "langgraph.toolnode"


def _recording(
    request,
    actor: Optional[str],
    control: Optional[dict[str, Any]],
    log: AuditLog,
    surface: str = SURFACE,
):
    """The recording arguments for one call, built in one place for both gates."""
    return {
        "audit": log,
        "surface": surface,
        "tool_use_id": request.tool_call["id"],
        "actor": actor,
        "control": control,
    }


def recusal_audited_gate(
    policy: Policy,
    log: AuditLog,
    *,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
    surface: str = SURFACE,
):
    """Return a synchronous ToolNode hook that records, then gates, each call.

    ``surface`` names the runtime on the record; another host of the same hook
    (LangChain's ``create_agent`` middleware) passes its own."""

    def gate(request, execute):
        refusal, judged = _judge(
            request, policy, **_recording(request, actor, control, log, surface)
        )
        return refusal if refusal is not None else execute(judged)

    return gate


def recusal_audited_agate(
    policy: Policy,
    log: AuditLog,
    *,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
    surface: str = SURFACE,
):
    """Return an asynchronous ToolNode hook for awrap_tool_call, same logic as the sync one."""

    async def gate(request, execute):
        refusal, judged = _judge(
            request, policy, is_async=True, **_recording(request, actor, control, log, surface)
        )
        return refusal if refusal is not None else await execute(judged)

    return gate
