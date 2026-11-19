"""Recusal for LangGraph: judge every ``ToolNode`` call before it runs.

    from recusal_langgraph import gated_tool_node

    node = gated_tool_node([run_sql], policy)

``policy(tool_name, tool_input)`` is an ordinary Recusal policy returning findings. A
refused call is answered with an error ``ToolMessage`` carrying Recusal's reason, and its
tool does not run. ``gated_tool_node`` puts the gate on both of ToolNode's hook slots, so
``invoke``, ``stream``, ``ainvoke`` and ``astream`` all cross it; ``recusal_gates(policy)``
is the same pair for a ToolNode you build yourself, and ``is_gated(node)`` checks one.
``audited_tool_node`` and ``recusal_audited_gates`` put each decision on a hash-chained
audit log.

The decision is made by ``recusal.decide``. This package adds only what is LangGraph's:
where the hook attaches, how the invocation is completed, and how a refusal is answered.
"""

from ._audit import (
    SURFACE,
    audited_tool_node,
    recusal_audited_agate,
    recusal_audited_gate,
    recusal_audited_gates,
)
from ._gate import (
    Policy,
    RecusalRefusal,
    gated_tool_node,
    is_gated,
    recusal_agate,
    recusal_gate,
    recusal_gates,
)

__version__ = "0.2.0"

__all__ = [
    "SURFACE",
    "Policy",
    "RecusalRefusal",
    "__version__",
    "audited_tool_node",
    "gated_tool_node",
    "is_gated",
    "recusal_agate",
    "recusal_audited_agate",
    "recusal_audited_gate",
    "recusal_audited_gates",
    "recusal_gate",
    "recusal_gates",
]
