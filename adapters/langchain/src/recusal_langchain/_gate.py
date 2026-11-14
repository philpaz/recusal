"""The gate as ``create_agent`` middleware: the LangGraph ``ToolNode`` gate, wrapped.

A LangChain ``create_agent`` agent executes its tools through a LangGraph ``ToolNode``, and
its tool-call middleware is the same ``wrap_tool_call`` seam. This package therefore adds no
decision logic of its own: ``gate_middleware(policy)`` is ``recusal_langgraph.recusal_gate``
handed to LangChain's ``wrap_tool_call``. Everything that gate judges, completes and refuses
applies here unchanged.

List the middleware LAST in ``middleware=``. The first middleware in the list is the
outermost, so anything listed after the gate runs between the gate and the tool: a
middleware that swaps the tool object or answers in the tool's place would act on a call
the gate never saw. Listed last, the gate judges the call exactly as it will run.
"""

from __future__ import annotations

from langchain.agents.middleware import wrap_tool_call
from recusal_langgraph import Policy, recusal_agate, recusal_gate


def gate_middleware(policy: Policy):
    """The gate as ``create_agent`` middleware, for ``agent.invoke``. List it LAST."""
    return wrap_tool_call(recusal_gate(policy))


def agate_middleware(policy: Policy):
    """The async gate, for ``agent.ainvoke``; the sync gate alone raises there. List it LAST."""
    return wrap_tool_call(recusal_agate(policy))
