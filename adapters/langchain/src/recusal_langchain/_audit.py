"""The audited middleware: the same decision, with every call on a hash-chained ``AuditLog``.

``audited_gate_middleware(policy, log)`` is one middleware object holding the audited sync
and async ``ToolNode`` hooks of ``recusal-langgraph``, which call the plain gate's own
judging function with the recording arguments passed through to ``recusal.decide``. There
is no second decision path. If the log cannot be written the decision becomes a deny, so
nothing runs unrecorded.

Each entry carries the surface ``langchain.create_agent`` and LangChain's tool call id, and
a SHA-256 fingerprint of the arguments, never the arguments themselves. It is bound to the
invocation the tool runs with, as the LangGraph audited gate binds it.

List the middleware LAST, as with the plain gate. The gate judges, and records, inside the
tool's own lifecycle, so an argument rewrite by a middleware listed after it is still
judged and recorded as rewritten. A middleware listed after it that swaps the tool object,
or answers in the tool's place, removes the judging copy: the gate withholds that answer,
refuses, and records the refusal as a call that was never judged, but a swapped-in tool has
run by then. One that answers a call the gate refused by running another tool is withheld
the same way, and the log then holds the refusal and the withheld answer, two entries.

Not on the log: a model call refused because it offers a provider-run tool. No tool call
exists at that point, so there is nothing to make an entry of.
"""

from __future__ import annotations

from typing import Any, Optional

from recusal_langgraph import Policy, recusal_audited_agate, recusal_audited_gate

from recusal.audit import AuditLog

from ._gate import _GateMiddleware

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
    """The audited gate as create_agent middleware, for ``invoke`` and ``ainvoke`` alike.
    List it LAST in ``middleware=``."""
    return _GateMiddleware(
        recusal_audited_gate(policy, log, actor=actor, control=control, surface=SURFACE),
        recusal_audited_agate(
            policy,
            log,
            actor=actor,
            control=control,
            surface=SURFACE,
            host_refuses_sync_runs=True,
        ),
    )


def audited_agate_middleware(
    policy: Policy,
    log: AuditLog,
    *,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
):
    """The same middleware as ``audited_gate_middleware``, kept under the name earlier
    releases asked for with ``agent.ainvoke``. One of the two is enough. List it LAST."""
    return audited_gate_middleware(policy, log, actor=actor, control=control)
