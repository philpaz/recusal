"""Recusal for LangChain ``create_agent``: judge every tool call before it runs.

    from langchain.agents import create_agent
    from recusal_langchain import gate_middleware

    agent = create_agent(model=model, tools=[run_sql], middleware=[gate_middleware(policy)])

``policy(tool_name, tool_input)`` is an ordinary Recusal policy returning findings. A
refused call is answered with an error ``ToolMessage`` carrying Recusal's reason, and its
tool does not run. ``agate_middleware`` is the form for ``agent.ainvoke``, and
``audited_gate_middleware`` and ``audited_agate_middleware`` put each decision on a
hash-chained audit log. List the middleware last.

The gate is ``recusal-langgraph``'s ``ToolNode`` gate, and the decision is made by
``recusal.decide``. This package adds only what is LangChain's: the middleware wrapper and
the surface label on the record.
"""

from ._audit import SURFACE, audited_agate_middleware, audited_gate_middleware
from ._gate import agate_middleware, gate_middleware

__version__ = "0.1.0"

__all__ = [
    "SURFACE",
    "__version__",
    "agate_middleware",
    "audited_agate_middleware",
    "audited_gate_middleware",
    "gate_middleware",
]
