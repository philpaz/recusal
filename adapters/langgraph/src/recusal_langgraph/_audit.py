"""The audited gate: the same decision, with every call on a hash-chained ``AuditLog``.

``recusal_audited_gate(policy, log)`` and the pair ``recusal_audited_gates(policy, log)``
call the plain gate's own judging function with the recording arguments passed through to
``recusal.decide``, which writes the entry before returning. There is no second decision
path. If the log cannot be written the decision becomes a deny, so nothing runs unrecorded.

Each entry carries the surface ``langgraph.toolnode`` and LangGraph's tool call id, and a
SHA-256 fingerprint of the arguments, never the arguments themselves. A call whose id is
empty or not a string is refused, and its entry carries no call id.

What each entry is bound to:

- a call that is judged: the invocation the tool executes, validated and completed, with
  the arguments the host filled left out;
- a call refused before judgment (a tool that overrides the standard lifecycle, input that
  does not validate, a parameter the schema omits, a call that names another tool than the
  one that would run, a call with no id): the arguments as proposed;
- a call the executor answered without the judging step (:func:`recusal_langgraph._gate._judge`):
  the arguments as proposed, with a deny. The tool may have run before that entry was
  written; the entry says the call was never judged, not that the tool did not run.

What is not on the log: a call another ``wrap_tool_call`` hook answered without reaching
the gate.
"""

from __future__ import annotations

from typing import Any, Optional

from langgraph.prebuilt import ToolNode

from recusal.audit import AuditLog

from ._gate import (
    Policy,
    _agated,
    _checked,
    _gated,
    _marked,
    _no_hooks_among,
    _require_sync_refusal,
    _require_sync_slot,
)

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
        _require_sync_slot("recusal_audited_gate", execute)
        record = _recording(request, actor, control, log, surface)
        return _gated(request, execute, policy, **record)

    return _marked(gate, "sync", policy, log)


def _audited_agate(policy, log, actor, control, surface):
    async def gate(request, execute):
        record = _recording(request, actor, control, log, surface)
        return await _agated(request, execute, policy, **record)

    return _marked(gate, "async", policy, log)


def recusal_audited_agate(
    policy: Policy,
    log: AuditLog,
    *,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
    surface: str = SURFACE,
    host_refuses_sync_runs: bool = False,
):
    """Return the asynchronous recording hook on its own, for a host that refuses a sync
    run itself. A ToolNode is not one (see ``recusal_agate``), so this raises ``TypeError``
    unless ``host_refuses_sync_runs=True``. For a ToolNode use ``recusal_audited_gates`` or
    ``audited_tool_node``."""
    _require_sync_refusal("recusal_audited_agate", host_refuses_sync_runs)
    return _audited_agate(policy, log, actor, control, surface)


def recusal_audited_gates(
    policy: Policy,
    log: AuditLog,
    *,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
    surface: str = SURFACE,
) -> dict[str, Any]:
    """Both recording hooks, keyed by the ToolNode parameter each one fills.

    node = ToolNode(tools, **recusal_audited_gates(policy, log))
    """
    return {
        "wrap_tool_call": recusal_audited_gate(
            policy, log, actor=actor, control=control, surface=surface
        ),
        "awrap_tool_call": _audited_agate(policy, log, actor, control, surface),
    }


def audited_tool_node(
    tools,
    policy: Policy,
    log: AuditLog,
    *,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
    **options: Any,
) -> ToolNode:
    """A ``ToolNode`` over ``tools`` with the recording gate on both of its hook slots.
    ``options`` are ToolNode's own; passing either hook slot raises ``TypeError``, and so
    does a tool ``gated_tool_node`` refuses to build with."""
    _no_hooks_among(options)
    return _checked(
        ToolNode(
            tools, **recusal_audited_gates(policy, log, actor=actor, control=control), **options
        )
    )
