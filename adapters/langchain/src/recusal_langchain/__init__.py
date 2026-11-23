"""Recusal for LangChain ``create_agent``: judge every tool call before it runs.

    from langchain.agents import create_agent
    from recusal_langchain import gate_middleware

    agent = create_agent(model=model, tools=[run_sql], middleware=[gate_middleware(policy)])

``policy(tool_name, tool_input)`` is an ordinary Recusal policy returning findings. A
refused call is answered with an error ``ToolMessage`` carrying Recusal's reason, and its
tool does not run. The one middleware serves ``agent.invoke`` and ``agent.ainvoke``;
``audited_gate_middleware`` puts each decision on a hash-chained audit log.
``agate_middleware`` and ``audited_agate_middleware`` are the same two middleware under
the names earlier releases used for ``ainvoke``. List the middleware last.

The gate is ``recusal-langgraph``'s ``ToolNode`` gate, and the decision is made by
``recusal.decide``. This package adds only what is LangChain's: the middleware object, the
refusal of a model call that offers the model a provider-run tool, and the surface label
on the record.
"""

from ._audit import SURFACE, audited_agate_middleware, audited_gate_middleware
from ._gate import agate_middleware, gate_middleware

__version__ = "0.2.0"

__all__ = [
    "SURFACE",
    "__version__",
    "agate_middleware",
    "audited_agate_middleware",
    "audited_gate_middleware",
    "gate_middleware",
]
