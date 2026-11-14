"""The audited gate: the same decision, with every call on a hash-chained ``AuditLog``.

``install_audited_hooks(policy, log)`` is ``install_recusal_hooks`` with a ``recording``
that passes the recording arguments through to ``recusal.decide``, which writes the entry
before returning. There is no second decision path. If the log cannot be written the
decision becomes a deny, so nothing runs unrecorded. Like the plain gate it is
process-global: one install records every agent, task and crew in the process.

What each entry carries:

- no call id. CrewAI's hook context has the tool name, the argument dict, the tool and the
  agent, task and crew, and no id for the call, so the record does not invent one. Entries
  are told apart by their order on the chain, the tool, the fingerprint and
  ``control.crewai_agent``, the role of the agent that made the call when CrewAI supplies
  it;
- for a judged call (allowed or refused by the policy): the fingerprint of the prepared
  invocation CrewAI executes, defaults completed and values validated;
- for a call refused before judgment (a tool or schema the gate cannot judge, an unknown or
  missing parameter, a validation failure): the fingerprint of the arguments as proposed.

What is not on the log: a call a hook registered BEFORE the gate blocked (CrewAI stops at
the first hook that blocks, so the gate never sees it and the tool does not run), and
whatever a hook registered AFTER the gate changed. Install the gate after your other hooks,
as with the plain one, so the record is of the call exactly as it runs.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from recusal.audit import AuditLog

from ._gate import Policy, Recording, install_recusal_hooks

#: The surface label every entry carries: the CrewAI hook that adjudicated.
SURFACE = "crewai.before_tool_call"


def recording(
    log: AuditLog, *, actor: Optional[str] = None, control: Optional[dict[str, Any]] = None
) -> Recording:
    """The recording arguments for each call, built in one place.

    CrewAI's hook context carries no call id, so none is recorded. The role of the agent
    making the call is recorded as ``control.crewai_agent`` when CrewAI supplies one; a
    caller's own ``control`` cannot overwrite it."""

    def for_call(context: Any) -> dict[str, Any]:
        identity = dict(control or {})
        role = getattr(getattr(context, "agent", None), "role", None)
        if isinstance(role, str) and role:
            identity["crewai_agent"] = role
        else:
            identity.pop("crewai_agent", None)
        return {
            "audit": log,
            "surface": SURFACE,
            "actor": actor,
            "control": identity or None,
        }

    return for_call


def install_audited_hooks(
    policy: Policy,
    log: AuditLog,
    *,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
) -> Callable[[], None]:
    """``install_recusal_hooks`` with every decision on ``log``; returns ``uninstall``."""
    return install_recusal_hooks(policy, recording(log, actor=actor, control=control))
