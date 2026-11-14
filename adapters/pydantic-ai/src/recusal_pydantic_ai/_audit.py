"""The audited gate: the same decision, with every call on a hash-chained ``AuditLog``.

``RecusalAuditedGate(policy, log=log)`` and ``recusal_audited_before_tool_execute(policy,
log)`` call the plain gate's own ``_refusal`` with the recording arguments passed through
to ``recusal.decide``, which writes the entry before returning. There is no second decision
path. If the log cannot be written the decision becomes a deny, so nothing runs unrecorded.

Each entry carries the surface ``pydantic_ai.before_tool_execute`` and Pydantic AI's own
tool call id, and a SHA-256 fingerprint of the arguments, never the arguments themselves.

What each entry is bound to:

- a call that is judged (allowed or refused by the policy): the validated arguments the
  tool is called with;
- a call refused before judgment (arguments the gate cannot detach): the call as the model
  proposed it, since the validated arguments are not data a fingerprint can cover.

What is not on the log, because the hook never runs for it: a call whose arguments fail
Pydantic AI's validation (answered with a retry prompt, and the tool does not run), and
the tool calls of an agent another agent delegates to, unless that agent has its own
audited gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from pydantic_ai.exceptions import ToolFailed
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

from recusal.audit import AuditLog

from ._gate import Policy, RecusalGate, _refusal

#: The surface label every entry carries: the runtime hook that adjudicated.
SURFACE = "pydantic_ai.before_tool_execute"


def _recording(
    call: ToolCallPart, actor: Optional[str], control: Optional[dict[str, Any]], log: AuditLog
) -> dict[str, Any]:
    """The recording arguments for one call, built in one place for both forms of the gate.

    Pydantic AI generates a tool call id when the model sends none, so an entry normally
    carries one; a blank id is recorded as no id rather than refused."""
    ident = call.tool_call_id
    return {
        "audit": log,
        "surface": SURFACE,
        "tool_use_id": ident if isinstance(ident, str) and ident else None,
        "actor": actor,
        "control": control,
    }


def _proposal(call: ToolCallPart) -> Any:
    """The call's arguments as the model sent them, for the entry of a call refused before
    judgment. Never raises: an argument string that is not a JSON object is kept as text."""
    try:
        return call.args_as_dict()
    except Exception:  # noqa: BLE001, the refusal is recorded whatever the proposal looks like
        return call.args if isinstance(call.args, (str, dict)) else None


def recusal_audited_before_tool_execute(
    policy: Policy,
    log: AuditLog,
    *,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
):
    """Return a ``before_tool_execute`` hook that records, then gates, each call.

    For ``Hooks(before_tool_execute=...)``, which declares no ordering: list it last, as
    with the plain hook, or a later hook can change the arguments after they were recorded.
    """

    async def gate(ctx: Any, *, call: ToolCallPart, tool_def: ToolDefinition, args: dict) -> dict:
        reason = _refusal(
            call.tool_name,
            args,
            policy,
            proposal=_proposal(call),
            **_recording(call, actor, control, log),
        )
        if reason is not None:
            raise ToolFailed(reason)
        return args

    gate.recusal_gate = True  # type: ignore[attr-defined]
    return gate


@dataclass(kw_only=True)
class RecusalAuditedGate(RecusalGate):
    """The audited gate as a capability: ``RecusalAuditedGate(policy, log=log)``.

    A ``RecusalGate`` (so ``is_gated`` sees it, and it keeps the ``innermost`` position)
    whose every decision is one entry on ``log``.
    """

    log: AuditLog
    actor: Optional[str] = None
    control: Optional[dict[str, Any]] = None

    async def before_tool_execute(
        self, ctx: Any, *, call: ToolCallPart, tool_def: ToolDefinition, args: dict
    ) -> dict:
        reason = _refusal(
            call.tool_name,
            args,
            self.policy,
            proposal=_proposal(call),
            **_recording(call, self.actor, self.control, self.log),
        )
        if reason is not None:
            raise ToolFailed(reason)
        return args
