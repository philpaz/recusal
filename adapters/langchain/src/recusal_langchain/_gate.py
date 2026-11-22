"""The gate as ``create_agent`` middleware: the LangGraph ``ToolNode`` gate, wrapped.

A LangChain ``create_agent`` agent executes its tools through a LangGraph ``ToolNode``, and
its tool-call middleware is the same ``wrap_tool_call`` seam. This package therefore adds no
decision logic of its own for a tool call: ``gate_middleware(policy)`` is one middleware
object holding ``recusal_langgraph``'s sync and async hooks, one for each way LangChain runs
a tool call, so the same agent is gated under ``invoke`` and under ``ainvoke``. Everything
that gate judges, completes and refuses applies here unchanged.

The one thing it adds is LangChain's: a ``create_agent`` agent can be given a tool as a
dict, which the model provider runs on its own side (a provider's web search or code
execution). No tool-call middleware sees such a call. The middleware is handed the tools
of every model call before the model is called, so it refuses the model call instead:
the model is not called, and the agent's answer is the refusal.

List the middleware LAST in ``middleware=``. The first middleware in the list is the
outermost, so anything listed after the gate runs between the gate and the tool: a
middleware that swaps the tool object or answers in the tool's place would act on a call
the gate never judged (the gate then withholds the answer and refuses, but a swapped-in
tool has already run), and one that adds a provider-run tool to the model call does so
after the gate looked. Listed last, the gate judges the call exactly as it will run.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from recusal_langgraph import Policy, recusal_agate, recusal_gate


def _provider_run(request: Any) -> Optional[AIMessage]:
    """The answer to a model call that offers the model a tool given as a dict, or None.

    ``create_agent`` keeps a dict in ``tools=`` out of its ``ToolNode`` and binds it to the
    model as it is: the provider runs it, and no ``wrap_tool_call`` hook is ever called for
    it. The refusal names how many there are and nothing from inside them.
    """
    unseen = sum(1 for tool in request.tools if isinstance(tool, dict))
    if not unseen:
        return None
    return AIMessage(
        content=(
            f"Recusal refused this model call: it offers the model {unseen} tool(s) given "
            "as a dict, which the model provider runs on its own side. No tool-call "
            "middleware sees such a call, so no policy could judge it. Remove the dict "
            "tool(s) from tools=, or give the agent a tool it runs itself in their place."
        )
    )


class _GateMiddleware(AgentMiddleware):
    """One middleware for both routes: LangChain calls ``wrap_tool_call`` for a tool call
    under ``invoke`` and ``stream``, and ``awrap_tool_call`` under ``ainvoke`` and
    ``astream``; each holds the ``recusal_langgraph`` hook for that route. The two model
    hooks refuse a model call that offers a provider-run tool (:func:`_provider_run`)."""

    tools = ()

    def __init__(self, sync_hook: Any, async_hook: Any) -> None:
        super().__init__()
        self._sync_hook, self._async_hook = sync_hook, async_hook

    @property
    def name(self) -> str:
        return "recusal_gate"

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        return self._sync_hook(request, handler)

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        return await self._async_hook(request, handler)

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        refusal = _provider_run(request)
        return refusal if refusal is not None else handler(request)

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        refusal = _provider_run(request)
        return refusal if refusal is not None else await handler(request)


def gate_middleware(policy: Policy):
    """The gate as ``create_agent`` middleware, for ``invoke`` and ``ainvoke`` alike. List
    it LAST."""
    return _GateMiddleware(recusal_gate(policy), recusal_agate(policy, host_refuses_sync_runs=True))


def agate_middleware(policy: Policy):
    """The same middleware as ``gate_middleware``, kept under the name earlier releases
    asked for with ``agent.ainvoke``. One of the two is enough. List it LAST."""
    return gate_middleware(policy)
