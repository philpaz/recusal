"""Recusal for LangGraph: judge every ``ToolNode`` call before it runs.

    from langgraph.prebuilt import ToolNode
    from recusal_langgraph import recusal_gate

    node = ToolNode([run_sql], wrap_tool_call=recusal_gate(policy))

``policy(tool_name, tool_input)`` is an ordinary Recusal policy returning findings. A
refused call is answered with an error ``ToolMessage`` carrying Recusal's reason, and its
tool does not run. ``recusal_agate`` is the ``awrap_tool_call`` form, and
``recusal_audited_gate`` and ``recusal_audited_agate`` put each decision on a hash-chained
audit log.

The decision is made by ``recusal.decide``. This package adds only what is LangGraph's:
where the hook attaches, how the invocation is completed, and how a refusal is answered.
"""

from ._audit import SURFACE, recusal_audited_agate, recusal_audited_gate
from ._gate import Policy, RecusalRefusal, recusal_agate, recusal_gate

__version__ = "0.1.0"

__all__ = [
    "SURFACE",
    "Policy",
    "RecusalRefusal",
    "__version__",
    "recusal_agate",
    "recusal_audited_agate",
    "recusal_audited_gate",
    "recusal_gate",
]
