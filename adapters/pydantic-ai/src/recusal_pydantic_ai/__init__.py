"""Recusal for Pydantic AI: judge every tool call before it runs.

    from pydantic_ai import Agent
    from recusal_pydantic_ai import RecusalGate

    agent = Agent(model, tools=[run_sql], capabilities=[RecusalGate(policy)])

``policy(tool_name, tool_input)`` is an ordinary Recusal policy returning findings. A
refused call is answered to the model as a failed tool result carrying Recusal's reason;
the tool does not run. ``RecusalAuditedGate(policy, log=AuditLog(path=...))`` is the same
gate with each decision on a hash-chained audit log.

The decision is made by ``recusal.decide``. This package adds only what is Pydantic AI's:
where the hook attaches, what it is handed, and how a refusal is answered.
"""

from ._audit import SURFACE, RecusalAuditedGate, recusal_audited_before_tool_execute
from ._gate import Policy, RecusalGate, is_gated, recusal_before_tool_execute

__version__ = "0.1.0"

__all__ = [
    "SURFACE",
    "Policy",
    "RecusalAuditedGate",
    "RecusalGate",
    "__version__",
    "is_gated",
    "recusal_audited_before_tool_execute",
    "recusal_before_tool_execute",
]
