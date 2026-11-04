"""
Offline LangGraph example: gate @tool calls with Recusal at ToolNode execution time.

Requires Python 3.10+ and is tested with exactly langgraph==1.2.12 and
langchain-core==1.6.5. Recusal itself remains Python 3.9+ and has no LangGraph dependency.

The hook is LangGraph-native: a user's existing @tool list goes unchanged into ToolNode and
recusal_gate(policy) is attached with wrap_tool_call. Every call in a model message crosses
the gate separately immediately before execution. Denials become error ToolMessages carrying
Recusal's reason; allow/defer execute normally. Async graphs attach recusal_agate(policy)
through awrap_tool_call and run with ainvoke, using the same decision logic.

No LLM, API key or database is used. run_sql only records SQL text.

Run after installing:
    pip install langgraph==1.2.12 langchain-core==1.6.5
    python examples/langgraph_gate.py
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, List

LANGGRAPH_VERSION = "1.2.12"
LANGCHAIN_CORE_VERSION = "1.6.5"

if sys.version_info < (3, 10):
    print(
        "langgraph_gate.py requires Python 3.10+ "
        f"(LangGraph {LANGGRAPH_VERSION}); "
        f"this is Python {sys.version_info[0]}.{sys.version_info[1]}.",
        file=sys.stderr,
    )
    raise SystemExit(2)

try:
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.tools import tool
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition
    from pydantic import BaseModel, field_validator
    from pydantic_core import to_jsonable_python
except ImportError as exc:
    raise SystemExit(
        "Install the example dependencies first: "
        f"langgraph=={LANGGRAPH_VERSION} langchain-core=={LANGCHAIN_CORE_VERSION}"
    ) from exc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.sql_scope_policy import policy as sql_scope_policy  # noqa: E402
from recusal import Finding  # noqa: E402
from recusal.claude_code import decide  # noqa: E402

Policy = Callable[[str, dict], List[Any]]
EXECUTED: list[tuple[str, dict[str, Any]]] = []


@tool
def echo(text: str) -> str:
    """Return harmless text."""
    EXECUTED.append(("echo", {"text": text}))
    return text


@tool
def run_sql(sql: str) -> str:
    """Record SQL text without connecting to a database."""
    EXECUTED.append(("run_sql", {"sql": sql}))
    return f"recorded SQL: {sql}"


class ChangedAfterJudgment(RuntimeError):
    """A model-facing argument differed, at execution, from the value the gate judged."""


def refuse_changed(request, exc: Exception, policy: Policy, **record: Any) -> ToolMessage:
    """The refusal for a call whose value changed after judgment: recorded through
    ``decide`` like every other refusal (so an audited gate logs it), returned to the model."""
    call = request.tool_call
    detail = str(exc)

    def changed(_name, _input):
        return [Finding.fail("recusal_changed_after_judgment", severity="CRITICAL", message=detail)]

    raw = call.get("args")
    try:
        _, reason = decide(
            call["name"], raw if isinstance(raw, dict) else {"input": raw}, changed, **record
        )
    except Exception as inner:  # noqa: BLE001, the refusal stands even if the record cannot be written
        reason = f"Recusal failed closed (policy error): {inner}"
    return ToolMessage(
        content=reason, tool_call_id=call.get("id"), name=call["name"], status="error"
    )


def _pinned(tool, judged: dict[str, Any], call_id):
    """A copy of the tool that executes only the invocation the gate judged.

    ``BaseTool.run`` validates the input through ``_to_args_and_kwargs`` and hands the result
    to ``_run``. The copy runs the tool's own validation on the input it is finally given
    (ToolNode injects host-supplied arguments, ``InjectedState`` and ``InjectedToolArg``,
    into the call after this hook returns) and then compares every model-facing argument
    (``tool.args``, the schema the model sees) with what was judged. A value that differs,
    because a later middleware rewrote the call or a validator is not idempotent, raises, so
    the tool does not run on an unjudged value. Injected arguments are the host's, never the
    model's, and pass through.
    """
    copy = tool.model_copy()
    original = type(tool)._to_args_and_kwargs
    model_facing = tuple(tool.args)

    def to_args_and_kwargs(tool_input, tool_call_id):
        if tool_call_id != call_id:
            raise RuntimeError("Recusal: a pinned tool was used for a different call")
        args, kwargs = original(copy, tool_input, tool_call_id)
        for key in model_facing:
            if to_jsonable_python(kwargs.get(key)) != judged.get(key):
                raise ChangedAfterJudgment(
                    f"`{key}` changed after it was judged; the call does not run"
                )
        return args, kwargs

    object.__setattr__(copy, "_to_args_and_kwargs", to_args_and_kwargs)
    return copy


def _judge(request, policy: Policy, **record: Any):
    """Judge the invocation the tool will execute; return ``(refusal, request_to_execute)``.

    The policy sees the arguments after the tool's own validation (``args_schema``: field
    validators, defaults, coercion), because that is what the tool receives, not the dict the
    model wrote. An allowed call comes back as a request whose tool is pinned to that
    validated invocation, so ``execute`` runs exactly what was judged. An input the tool
    rejects, or a call to a tool the node does not know, is refused through ``decide`` as
    well, so it lands on the audit log like every other decision. ``record`` goes straight
    to ``decide`` (``audit=``, ``surface=``, ``tool_use_id=``, ...), so a gate that puts the
    decision on a log (examples/langgraph_audit.py) decides, and refuses, exactly as this one
    does: one function, no second copy to drift.
    """
    call = request.tool_call
    name, call_id, raw = call["name"], call.get("id"), call.get("args")

    def refuse(reason):
        return ToolMessage(content=reason, tool_call_id=call_id, name=name, status="error"), None

    def refuse_unjudgeable(detail):
        def unjudgeable(_name, _input):
            return [Finding.fail("recusal_unjudgeable_input", severity="CRITICAL", message=detail)]

        bound = raw if isinstance(raw, dict) else {"input": raw}
        try:
            _, reason = decide(name, bound, unjudgeable, **record)
        except Exception as exc:
            reason = f"Recusal failed closed (policy error): {exc}"
        return refuse(reason)

    tool = request.tool
    if tool is None:
        return refuse_unjudgeable("no tool by that name is registered with this node")
    try:
        args, kwargs = tool._to_args_and_kwargs(raw, call_id)
    except Exception as exc:  # noqa: BLE001, an input the tool rejects is refused with its reason
        return refuse_unjudgeable(f"arguments do not validate ({exc})")
    if args:
        return refuse_unjudgeable("a positional tool input cannot be judged by name")
    try:
        effective = {key: to_jsonable_python(value) for key, value in kwargs.items()}
    except Exception as exc:  # noqa: BLE001, a value the record cannot bind is refused
        return refuse_unjudgeable(f"cannot read its validated arguments ({exc})")
    try:
        decision, reason = decide(name, effective, policy, **record)
    except Exception as exc:
        decision, reason = "deny", f"Recusal failed closed (policy error): {exc}"
    if decision == "deny":
        return refuse(reason)
    return None, request.override(tool=_pinned(tool, effective, call_id))


def recusal_gate(policy: Policy):
    """Return a synchronous ToolNode hook that gates each call before execution."""

    def gate(request, execute):
        refusal, pinned = _judge(request, policy)
        if refusal is not None:
            return refusal
        try:
            return execute(pinned)
        except ChangedAfterJudgment as exc:
            return refuse_changed(request, exc, policy)

    return gate


def recusal_agate(policy: Policy):
    """Return an asynchronous ToolNode hook for awrap_tool_call."""

    async def gate(request, execute):
        refusal, pinned = _judge(request, policy)
        if refusal is not None:
            return refusal
        try:
            return await execute(pinned)
        except ChangedAfterJudgment as exc:
            return refuse_changed(request, exc, policy)

    return gate


def tool_message(calls: list[dict[str, Any]]) -> AIMessage:
    """A scripted model message proposing tool calls; no model is invoked."""
    return AIMessage(content="", tool_calls=calls)


def _graph(node: ToolNode, calls: list[dict[str, Any]]):
    """Build the same scripted agent graph for sync and async invocation."""
    script = iter([tool_message(calls)])

    def scripted_agent(state: MessagesState) -> dict[str, Any]:
        return {"messages": [next(script, AIMessage(content="done"))]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", scripted_agent)
    builder.add_node("tools", node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile()


def run_in_graph(node: ToolNode, calls: list[dict[str, Any]]) -> list[ToolMessage]:
    """Run a ToolNode inside the MessagesState graph used by an agent."""
    messages = _graph(node, calls).invoke({"messages": []})["messages"]
    return [message for message in messages if isinstance(message, ToolMessage)]


async def run_in_graph_async(node: ToolNode, calls: list[dict[str, Any]]) -> list[ToolMessage]:
    """Run a ToolNode through ainvoke, exercising its async hook and async tools."""
    messages = (await _graph(node, calls).ainvoke({"messages": []}))["messages"]
    return [message for message in messages if isinstance(message, ToolMessage)]


class _RewritingSql(BaseModel):
    """An ``args_schema`` whose validator rewrites the SQL: the probe from the 2026-11-03
    review. The gate judges the rewritten value, because that is what the tool receives."""

    sql: str

    @field_validator("sql")
    @classmethod
    def rewrite(cls, value: str) -> str:
        return "DELETE FROM audit_log"


def transforming_run_sql():
    """``run_sql`` behind the rewriting schema."""

    @tool("run_sql", args_schema=_RewritingSql)
    def rewriting_run_sql(sql: str) -> str:
        """Record SQL text without connecting to a database."""
        EXECUTED.append(("run_sql", {"sql": sql}))
        return f"recorded SQL: {sql}"

    return rewriting_run_sql


def run_calls(
    calls: list[dict[str, Any]],
    *,
    policy: Policy = sql_scope_policy,
    gated: bool = True,
    tools: list[Any] | None = None,
) -> tuple[list[ToolMessage], list[tuple[str, dict[str, Any]]]]:
    EXECUTED.clear()
    tools = [echo, run_sql] if tools is None else tools
    node = ToolNode(tools, wrap_tool_call=recusal_gate(policy)) if gated else ToolNode(tools)
    return run_in_graph(node, calls), list(EXECUTED)


async def run_calls_async(
    calls: list[dict[str, Any]],
    *,
    policy: Policy = sql_scope_policy,
    gated: bool = True,
    tools: list[Any] | None = None,
) -> tuple[list[ToolMessage], list[tuple[str, dict[str, Any]]]]:
    EXECUTED.clear()
    tools = [echo, run_sql] if tools is None else tools
    node = ToolNode(tools, awrap_tool_call=recusal_agate(policy)) if gated else ToolNode(tools)
    return await run_in_graph_async(node, calls), list(EXECUTED)


def main() -> None:
    calls = [
        {"name": "echo", "args": {"text": "hello"}, "id": "1", "type": "tool_call"},
        {
            "name": "run_sql",
            "args": {"sql": "DELETE FROM audit_log"},
            "id": "2",
            "type": "tool_call",
        },
        {
            "name": "run_sql",
            "args": {"sql": "SELECT * FROM audit_log"},
            "id": "3",
            "type": "tool_call",
        },
    ]
    messages, executed = run_calls(calls)
    for message in messages:
        print(f"{message.name} [{message.status}]: {message.content}")
    print(f"tools that ran: {len(executed)}")


if __name__ == "__main__":
    main()
