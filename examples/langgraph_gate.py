"""
Offline LangGraph example: gate @tool calls with Recusal at ToolNode execution time.

Requires Python 3.10+ and is tested with exactly langgraph==1.2.12 and
langchain-core==1.6.5. Recusal itself remains Python 3.9+ and has no LangGraph dependency.

The hook is LangGraph-native: a user's existing @tool list goes unchanged into ToolNode and
recusal_gate(policy) is attached with wrap_tool_call. Every call in a model message crosses
the gate separately immediately before execution. Denials become error ToolMessages carrying
Recusal's reason; allow/defer execute normally. Async graphs attach recusal_agate(policy)
through awrap_tool_call and run with ainvoke, using the same decision logic.

The policy judges the invocation the tool executes, not the dict the model wrote: the hook
hands execution a per-call copy of the tool whose parse step (``_to_args_and_kwargs``, run by
``BaseTool.run``/``arun`` after ToolNode has injected the host's state, store and runtime)
validates the input once, completes any schema parameter the framework's parse dropped
(``_parse_input`` selects the kwargs it returns through the argument model's own
``model_dump``, so a field with ``exclude=True`` or a root serializer can vanish) from the
callable's default, projects every model-controlled argument through a projection that runs
no serializer of its own, decides, and on allow passes those same completed objects to the
function. Host arguments are excluded from judgment and can never be supplied by the model;
a tool whose class or instance overrides the standard lifecycle is refused rather than judged
in the wrong place.

No LLM, API key or database is used. run_sql only records SQL text.

Run after installing:
    pip install langgraph==1.2.12 langchain-core==1.6.5
    python examples/langgraph_gate.py
"""

from __future__ import annotations

import dataclasses
import datetime
import decimal
import enum
import inspect
import math
import os
import sys
import uuid
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
    from langchain_core.tools import BaseTool, StructuredTool, Tool, ToolException, tool
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition
    from langgraph.prebuilt.tool_node import _get_all_injected_args
    from pydantic import BaseModel, ValidationError, field_validator
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


class RecusalRefusal(ToolException):
    """Raised inside a judging copy's parse step: the call is refused and the message is the
    reason. ``BaseTool.run``/``arun`` turn it into an error ToolMessage through the copy's
    ``handle_tool_error``, which handles only this type and leaves every other
    ``ToolException`` to the original tool's own setting."""


# The execution lifecycle this gate relies on: ``invoke``/``ainvoke`` call ``run``/``arun``,
# which parse the input through ``_to_args_and_kwargs`` and then call ``_run``/``_arun``.
# The judging copy replaces that parse step, so a tool that overrides any of these on its
# class (other than the three standard implementations) or on the instance may execute
# without ever reaching the seam; such a tool is refused rather than judged in the wrong place.
_LIFECYCLE = ("invoke", "ainvoke", "run", "arun", "_to_args_and_kwargs")
_STANDARD = {
    name: {vars(cls)[name] for cls in (BaseTool, StructuredTool, Tool) if name in vars(cls)}
    for name in _LIFECYCLE
}


def _lifecycle_refusal(tool) -> str | None:
    """Why this tool's execution path cannot be judged at the parse seam, or None."""
    for name in _LIFECYCLE:
        if name in vars(tool):
            return f"`{name}` is overridden on the tool instance"
        if getattr(type(tool), name, None) not in _STANDARD[name]:
            return f"{type(tool).__name__} overrides `{name}`"
    return None


def _host_keys(tool) -> frozenset[str]:
    """The argument names the host supplies, never the model.

    ToolNode strips every ``InjectedToolArg``-annotated key from the model's arguments and
    adds back only the state, store and runtime it owns (``_inject_tool_args``), and
    ``_parse_input`` sets an ``InjectedToolCallId`` from the call id; a value under one of
    these keys at the parse seam is therefore the host's. Everything else the parse step
    produces, including a field an ``extra="allow"`` schema accepted, is model-controlled and
    is judged.
    """
    injected = _get_all_injected_args(tool)
    return frozenset(injected.all_injected_keys) | frozenset(
        getattr(tool, "_injected_args_keys", ())
    )


def _schema_fields(tool) -> frozenset[str]:
    """The parameter names the tool's ``args_schema`` declares (host-injected ones included;
    they are excluded elsewhere), or none for a tool without a model schema."""
    schema = tool.args_schema
    if isinstance(schema, dict):
        return frozenset(schema.get("properties", {}))
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return frozenset(schema.model_fields)
    return frozenset()


def _callable_of(tool):
    """The function the tool finally calls: ``func`` or ``coroutine`` for a StructuredTool,
    else the class's ``_run``."""
    func = getattr(tool, "func", None) or getattr(tool, "coroutine", None)
    return func if func is not None else type(tool)._run


def _completed(kwargs: dict[str, Any], tool, host_keys: frozenset[str]) -> dict[str, Any]:
    """The parsed kwargs with every suppressed schema parameter restored from the callable.

    ``_parse_input`` returns only the fields present in the validated model's
    ``model_dump()``, so a schema field with ``exclude=True`` or a model-wide serializer that
    omits it never reaches the kwargs, and the function would fill its own Python default
    after the policy judged an invocation without it. Each parameter the schema declares, the
    host does not inject, and the parse did not return is taken from the callable's default
    here, and passed explicitly, so the judged invocation is the executed one; a suppressed
    parameter with no default raises ``TypeError`` and the call is refused as unjudgeable.
    Parameters the framework fills at call time (``run_manager``, ``callbacks``, the config
    parameter) are not schema fields and are untouched.
    """
    fields = _schema_fields(tool)
    try:
        parameters = inspect.signature(_callable_of(tool)).parameters
    except (TypeError, ValueError) as exc:
        raise TypeError(f"the tool's callable cannot be inspected for its parameters ({exc})")
    completed = dict(kwargs)
    for name, parameter in parameters.items():
        if name in completed or name not in fields or name in host_keys:
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if parameter.default is parameter.empty:
            raise TypeError(
                f"the schema suppressed `{name}` and the callable has no default for it, so "
                "the invocation cannot be judged"
            )
        completed[name] = parameter.default
    return completed


def _faithful(value: Any, _stack: tuple[int, ...] = ()) -> Any:
    """A detached, serializer-free data projection of one parsed argument, for judgment.

    The policy receives this projection; the tool receives the original parsed objects. The
    projection reads a Pydantic model field by field (``getattr`` over ``model_fields`` plus
    ``model_extra``), so no ``field_serializer`` or custom ``model_dump`` runs: a serializer
    that displays ``SELECT`` for a field holding ``DELETE`` is exactly what this bypasses.
    Supported, each with a fixed meaning: ``None``, ``bool``, ``int``, ``str``; a finite
    ``float``; ``Enum`` by its value (projected in turn); ``datetime``/``date``/``time`` by
    ``isoformat()``; a finite ``Decimal`` and a ``UUID`` by ``str()``; ``list`` and ``tuple``
    as a new list; ``dict`` with ``str`` keys as a new dict; Pydantic v2 models and
    dataclasses as a new dict of their fields. Anything else (a Pydantic v1 model, a set, an
    arbitrary object, a non-finite number, a cyclic value) raises ``TypeError`` and the call is
    refused: a value that cannot be represented faithfully cannot be judged. Containers are
    new objects, so a policy that mutates what it was given cannot change the invocation.
    """
    if isinstance(value, enum.Enum):
        return _faithful(value.value, _stack)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("a non-finite float cannot be judged")
        return value
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        if not value.is_finite():
            raise TypeError("a non-finite Decimal cannot be judged")
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if id(value) in _stack:
        raise TypeError("a cyclic value cannot be judged")
    stack = _stack + (id(value),)
    if isinstance(value, (list, tuple)):
        return [_faithful(item, stack) for item in value]
    if isinstance(value, dict):
        projected = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"a mapping keyed by {type(key).__name__} cannot be judged")
            projected[key] = _faithful(item, stack)
        return projected
    if isinstance(value, BaseModel):
        projected = {
            name: _faithful(getattr(value, name), stack) for name in type(value).model_fields
        }
        for key, item in (value.model_extra or {}).items():
            projected[key] = _faithful(item, stack)
        return projected
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _faithful(getattr(value, field.name), stack)
            for field in dataclasses.fields(value)
        }
    raise TypeError(f"a {type(value).__name__} value cannot be judged faithfully")


def _sanitized(exc: Exception, host_keys: frozenset[str]) -> str:
    """A validation failure as location and message only: never the input, and for a
    host-supplied argument (whose value may be graph state) only the error type."""
    if isinstance(exc, ValidationError):
        parts = []
        for error in exc.errors(include_url=False, include_context=False, include_input=False):
            loc = tuple(str(part) for part in error.get("loc", ()))
            if loc and loc[0] in host_keys:
                parts.append(f"{'.'.join(loc)}: {error.get('type')}")
            else:
                parts.append(f"{'.'.join(loc) or 'input'}: {error.get('msg')}")
        return "; ".join(parts)
    return type(exc).__name__


def _unjudgeable(name: str, detail: str, bound: dict, policy: Policy, record: dict) -> str:
    """The reason for a call that cannot be judged, recorded through ``decide`` like every
    other refusal (so an audited gate logs it)."""

    def unjudgeable(_name, _input):
        return [Finding.fail("recusal_unjudgeable_input", severity="CRITICAL", message=detail)]

    try:
        _, reason = decide(name, bound, unjudgeable, **record)
    except Exception as exc:  # noqa: BLE001, the refusal stands even if the record cannot be written
        reason = f"Recusal failed closed (policy error): {exc}"
    return reason


def _judging(tool, call: dict[str, Any], policy: Policy, record: dict[str, Any]):
    """A per-call copy of ``tool`` that judges, then runs, exactly the invocation it parsed.

    ``BaseTool.run``/``arun`` parse the input through ``_to_args_and_kwargs`` and hand the
    result to ``_run``/``_arun``; ToolNode injects the host's state, store and runtime into
    the input before that (``_inject_tool_args``), whatever hook or middleware ran earlier.
    The copy's parse step therefore sees the final input: it runs the tool's own parser once,
    completes any schema parameter that parse suppressed from the callable's default
    (:func:`_completed`), projects every model-controlled argument with :func:`_faithful`,
    calls ``decide`` once, and on allow returns the very objects it judged, so the tool
    executes what was judged with no second validation and nothing to compare. A deny, an input the parser rejects,
    a positional input, or a value that cannot be projected raises :class:`RecusalRefusal`,
    which the copy's ``handle_tool_error`` turns into the error ToolMessage the model reads;
    any other ``ToolException`` keeps the original tool's handling. The copy serves one call
    id; every execution attempt of that call (a retry, a resume) is judged again.
    """
    name, call_id, fallback = call["name"], call.get("id"), call.get("args")
    copy = tool.model_copy()
    parse = type(tool)._to_args_and_kwargs
    host_keys = _host_keys(tool)
    original_handler = tool.handle_tool_error

    def proposal(tool_input) -> dict:
        """The model-facing input, for the record of a refusal that judged nothing."""
        if isinstance(tool_input, dict):
            try:
                return _faithful({k: v for k, v in tool_input.items() if k not in host_keys})
            except TypeError:
                pass
        return fallback if isinstance(fallback, dict) else {"input": fallback}

    def seam(tool_input, tool_call_id):
        if tool_call_id != call_id:
            raise RuntimeError("Recusal: a judging copy of a tool was used for a different call")
        try:
            args, kwargs = parse(copy, tool_input, tool_call_id)
        except Exception as exc:  # noqa: BLE001, an input the tool rejects is refused with its reason
            detail = f"arguments do not validate ({_sanitized(exc, host_keys)})"
            raise RecusalRefusal(_unjudgeable(name, detail, proposal(tool_input), policy, record))
        if args:
            detail = "a positional tool input cannot be judged by name"
            raise RecusalRefusal(_unjudgeable(name, detail, proposal(tool_input), policy, record))
        try:
            kwargs = _completed(kwargs, tool, host_keys)
            judged = {k: _faithful(v) for k, v in kwargs.items() if k not in host_keys}
        except TypeError as exc:
            raise RecusalRefusal(_unjudgeable(name, str(exc), proposal(tool_input), policy, record))
        try:
            decision, reason = decide(name, judged, policy, **record)
        except Exception as exc:  # noqa: BLE001, a buggy policy must not silently disable the gate
            decision, reason = "deny", f"Recusal failed closed (policy error): {exc}"
        if decision == "deny":
            raise RecusalRefusal(reason)
        return args, kwargs

    def handle(exc):
        if isinstance(exc, RecusalRefusal):
            return str(exc)
        if not original_handler:
            raise exc
        if original_handler is True:
            return str(exc)
        if isinstance(original_handler, str):
            return original_handler
        return original_handler(exc)

    object.__setattr__(copy, "_to_args_and_kwargs", seam)
    object.__setattr__(copy, "handle_tool_error", handle)
    return copy


def _judge(request, policy: Policy, **record: Any):
    """Return ``(refusal, request_to_execute)``: a refusal ToolMessage for a call that can
    never reach the seam, else the request with its tool replaced by the judging copy.

    ``record`` goes straight to ``decide`` (``audit=``, ``surface=``, ``tool_use_id=``, ...),
    so a gate that puts the decision on a log (examples/langgraph_audit.py) decides, and
    refuses, exactly as this one does: one function, no second copy to drift.
    """
    call = request.tool_call
    name, call_id, raw = call["name"], call.get("id"), call.get("args")

    def refuse(detail):
        bound = raw if isinstance(raw, dict) else {"input": raw}
        reason = _unjudgeable(name, detail, bound, policy, record)
        return ToolMessage(content=reason, tool_call_id=call_id, name=name, status="error"), None

    tool = request.tool
    if tool is None:
        return refuse("no tool by that name is registered with this node")
    why = _lifecycle_refusal(tool)
    if why is not None:
        return refuse(
            f"the tool's execution path cannot be judged ({why}); the gate judges inside the "
            "standard BaseTool lifecycle, so build the tool with @tool or StructuredTool"
        )
    return None, request.override(tool=_judging(tool, call, policy, record))


def recusal_gate(policy: Policy):
    """Return a synchronous ToolNode hook that gates each call before execution."""

    def gate(request, execute):
        refusal, judged = _judge(request, policy)
        return refusal if refusal is not None else execute(judged)

    return gate


def recusal_agate(policy: Policy):
    """Return an asynchronous ToolNode hook for awrap_tool_call."""

    async def gate(request, execute):
        refusal, judged = _judge(request, policy)
        return refusal if refusal is not None else await execute(judged)

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
