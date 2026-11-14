"""The audited middleware: the same decision, with every call on a hash-chained ``AuditLog``.

``audited_gate_middleware(policy, log)`` and ``audited_agate_middleware(policy, log)`` wrap
the audited ``ToolNode`` hooks of ``recusal-langgraph``, which call the plain gate's own
judging function with the recording arguments passed through to ``recusal.decide``. There
is no second decision path. If the log cannot be written the decision becomes a deny, so
nothing runs unrecorded.

Each entry carries the surface ``langchain.create_agent`` and LangChain's tool call id, and
a SHA-256 fingerprint of the arguments, never the arguments themselves. It is bound to the
invocation the tool runs with, as the LangGraph audited gate binds it.

List the middleware LAST, as with the plain gate. The gate judges, and records, inside the
tool's own lifecycle, so an argument rewrite by a middleware listed after it is still
judged and recorded as rewritten; but a middleware listed after it that swaps the tool
object, or answers in the tool's place, removes the judging copy, and that call is neither
judged nor recorded.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain.agents.middleware import wrap_tool_call
from recusal_langgraph import Policy, recusal_audited_agate, recusal_audited_gate

from recusal.audit import AuditLog

#: The surface label every entry carries: the runtime that adjudicated, a ``create_agent``
#: agent's tool-call middleware.
SURFACE = "langchain.create_agent"


def audited_gate_middleware(
    policy: Policy,
    log: AuditLog,
    *,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
):
    """The audited gate as create_agent middleware. List it LAST in ``middleware=``."""
    return wrap_tool_call(
        recusal_audited_gate(policy, log, actor=actor, control=control, surface=SURFACE)
    )


def audited_agate_middleware(
    policy: Policy,
    log: AuditLog,
    *,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
):
    """The audited async gate for ``agent.ainvoke``. List it LAST."""
    return wrap_tool_call(
        recusal_audited_agate(policy, log, actor=actor, control=control, surface=SURFACE)
    )
