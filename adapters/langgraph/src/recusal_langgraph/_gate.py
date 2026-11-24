"""The gate: judge each ``ToolNode`` call at execution time, on the invocation that runs.

The hook is LangGraph-native: a user's existing ``@tool`` list goes unchanged into
``ToolNode`` and ``recusal_gate(policy)`` is attached with ``wrap_tool_call``. Every call in
a model message crosses the gate separately immediately before execution. Denials become
error ``ToolMessage`` objects carrying Recusal's reason; allow and defer execute normally.

A ``ToolNode`` has two hook slots and consults them by route. Its sync route (``invoke``,
``stream``) looks only at ``wrap_tool_call``: a node given only ``awrap_tool_call`` runs
every tool there with no hook at all. So the gate is attached to BOTH slots, by
``gated_tool_node(tools, policy)`` or ``ToolNode(tools, **recusal_gates(policy))``, and the
async hook cannot be built on its own except for a host that refuses a sync run itself
(``recusal_agate(policy, host_refuses_sync_runs=True)``). ``is_gated(node)`` reads a node
and says whether every route through it crosses the gate.

The copy's callable is guarded as well: it runs only for an invocation the parse step just
judged and allowed, so a route that reached the tool body without the parse step (a
framework release that changes the lifecycle) is refused instead of run unjudged. And the
hook passes an answer on only when it fits what the copy's parse step last did for the call
(:func:`_judge`): when that step never ran, when it refused and the answer is not that
refusal, or when the answer carries another tool's name, the hook refuses instead, saying
that a tool may have run.

The policy judges the invocation the tool executes, not the dict the model wrote: the hook
hands execution a per-call copy of the tool whose parse step (``_to_args_and_kwargs``, run by
``BaseTool.run``/``arun`` after ToolNode has injected the host's state, store and runtime)
validates the input once, holds every callable the route reaches to the schema (every named
parameter the function, or a wrapper around it, declares is a field the schema declares, a
key the host filled, or a name the framework reserves and fills at call time; any other is a
default the model's proposal cannot carry and the policy would never see, so the call is
refused; and when the innermost callable takes ``*args`` or ``**kwargs``, every schema field
must be one of its parameters, or it hands arguments on to code the gate cannot read and the
call is refused; the one exception is the function ``langchain-mcp-adapters`` defines to
call an MCP server, :func:`_calls_an_mcp_server`), completes any schema parameter the framework's parse dropped (``_parse_input``
selects the kwargs it returns through the argument model's own ``model_dump``, so a field
with ``exclude=True`` or a root serializer can vanish) from the default that would run,
projects every model-controlled argument through a projection that runs no serializer of its
own, decides, and on allow passes those same completed objects to the function. Host
arguments are excluded from judgment, and an argument counts as the host's only when ToolNode
put it in the input for this call: a value the model sent that arrives unchanged is judged,
whatever the tool's annotations say, and so is a value the parse step fills itself under an
injected name (the call id, which the model writes, or a schema default). A tool whose class
or instance overrides the standard lifecycle is refused rather than judged in the wrong
place; so is a call whose name is not the name of the tool that will run.
"""

from __future__ import annotations

import dataclasses
import datetime
import decimal
import enum
import functools
import inspect
import math
import sys
import typing
import uuid
import weakref
from typing import Any, Callable, List

from langchain_core.callbacks import BaseCallbackManager
from langchain_core.callbacks.manager import BaseRunManager
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import run_in_executor
from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool, Tool, ToolException
from langchain_core.tools.base import FILTERED_ARGS, _get_runnable_config_param
from langgraph.prebuilt import ToolNode, ToolRuntime
from langgraph.prebuilt.tool_node import _get_all_injected_args
from pydantic import BaseModel, ValidationError

import recusal.invocation as _invocation
from recusal import Finding, decide
from recusal.invocation import DeclaredSignature as _DeclaredSignature
from recusal.invocation import reached_parameters as _reached_parameters

#: A Recusal policy: ``policy(tool_name, tool_input)`` returns a list of findings.
Policy = Callable[[str, dict], List[Any]]


class _NotInThisKernel(Exception):
    """Stands in for a refusal of the walk that the installed kernel does not have.
    ``recusal`` 0.15.0, the oldest this package allows, has neither ``HiddenWrapper`` nor
    ``DispatchedAtCallTime``. Nothing raises this class, so a handler for it never runs."""


def _walk_refusal(kernel: Any, name: str) -> type[BaseException]:
    """The exception class the kernel's walk raises under ``name``, or the stand-in where
    the installed kernel has none of that name."""
    found = getattr(kernel, name, None)
    if isinstance(found, type) and issubclass(found, BaseException):
        return found
    return _NotInThisKernel


#: A callable on the way holds a second wrapper the walk does not read (a kernel later
#: than 0.15.1).
_HiddenWrapper = _walk_refusal(_invocation, "HiddenWrapper")
#: A single-dispatch function on the way can run more than its base (the same kernels).
_DispatchedAtCallTime = _walk_refusal(_invocation, "DispatchedAtCallTime")
#: A callable on the way is read by ``inspect.signature`` as a ``functools.partialmethod``
#: (the same kernels).
_PartialMethodSignature = _walk_refusal(_invocation, "PartialMethodSignature")
#: A bound argument is taken by a wrapper under another name than the parameter with a
#: default it would fill on the function the wrapper names (the same kernels).
_BoundArgumentKept = _walk_refusal(_invocation, "BoundArgumentKept")
#: A cache on the way names one function and calls another that cannot be told (the same
#: kernels). A ``HiddenWrapper`` there, so it is asked for first.
_UnreadCache = _walk_refusal(_invocation, "UnreadCache")


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


# A ``StructuredTool`` or ``Tool`` runs the function it holds (``func``/``coroutine``)
# from its own ``_run``/``_arun``, and the gate holds the invocation to that function's
# signature. A subclass that replaces ``_run`` or ``_arun`` executes something else, which
# the policy's view of the invocation was never checked against, so it is refused. (A
# direct ``BaseTool`` subclass has no such function: its ``_run`` is the tool, and is the
# callable the gate inspects.)
_HOLDER_BODIES = {
    name: {vars(cls)[name] for cls in (StructuredTool, Tool) if name in vars(cls)}
    for name in ("_run", "_arun")
}


def _lifecycle_refusal(tool) -> str | None:
    """Why this tool's execution path cannot be judged at the parse seam, or None."""
    for name in _LIFECYCLE:
        if name in vars(tool):
            return f"`{name}` is overridden on the tool instance"
        if getattr(type(tool), name, None) not in _STANDARD[name]:
            return f"{type(tool).__name__} overrides `{name}`"
    if isinstance(tool, (StructuredTool, Tool)):
        for name, standard in _HOLDER_BODIES.items():
            if name in vars(tool):
                return f"`{name}` is overridden on the tool instance"
            if getattr(type(tool), name, None) not in standard:
                return f"{type(tool).__name__} overrides `{name}`"
    return None


def _host_keys(tool) -> frozenset[str]:
    """The argument names the host supplies, never the model.

    ToolNode strips every ``InjectedToolArg``-annotated key from the model's arguments and
    adds back only the state, store and runtime it owns (``_inject_tool_args``); a value
    under one of these keys in the input that reaches the parse seam is therefore the
    host's. A key the parse step fills itself is not: ``_parse_input`` sets an
    ``InjectedToolCallId`` from the call id, which the model writes, and gives a key the host
    left out its schema default. Those, and everything else the parse step produces,
    including a field an ``extra="allow"`` schema accepted, are judged.

    Both names read here are private to the frameworks. If either is missing or raises, the
    call is refused (``TypeError``): there is no fallback to "no host keys".
    """
    try:
        injected = _get_all_injected_args(tool)
        own = tool._injected_args_keys
    except Exception as exc:  # noqa: BLE001, without them the model's arguments cannot be told apart
        raise TypeError(
            "the arguments the host injects into the tool cannot be read from the installed "
            f"LangChain and LangGraph ({type(exc).__name__}), so the model's arguments cannot "
            "be told from the host's"
        ) from None
    return frozenset(injected.all_injected_keys) | frozenset(own)


def _node_fills(tool) -> frozenset[str]:
    """The argument names ToolNode itself puts a value under: the tool's state, store and
    runtime parameters.

    ``_inject_tool_args`` runs after every hook and middleware. It removes every key it
    counts as injected and adds back exactly these. So a value that reaches the parse seam
    under one of these names is the host's, whatever a hook wrote there; and a value that
    arrives under any other injected name (:func:`_host_keys`) was not removed by ToolNode,
    so nothing of the host's replaced it: it is the model's, or a hook's, and it is judged.

    Read from the same private helper as :func:`_host_keys`, and refused the same way when
    it cannot be read.
    """
    try:
        injected = _get_all_injected_args(tool)
        names = {*injected.state, injected.store, injected.runtime}
    except Exception as exc:  # noqa: BLE001, without them the model's arguments cannot be told apart
        raise TypeError(
            "the arguments the host injects into the tool cannot be read from the installed "
            f"LangChain and LangGraph ({type(exc).__name__}), so the model's arguments cannot "
            "be told from the host's"
        ) from None
    return frozenset(name for name in names if name)


def _schema_fields(tool) -> frozenset[str]:
    """The parameter names the tool's ``args_schema`` declares (host-injected ones included;
    they are excluded elsewhere). A tool that declares no schema (a ``BaseTool`` subclass
    with only ``_run``) has the one LangChain infers from ``_run``, read through the public
    ``get_input_schema``. A schema that is neither a JSON schema dict nor a Pydantic v2
    model (a Pydantic v1 model) declares nothing this gate can read, so the call is refused
    (``TypeError``): no callable could be held to it."""
    schema = tool.args_schema
    if schema is None:
        try:
            schema = tool.get_input_schema()
        except Exception as exc:  # noqa: BLE001, no schema to hold the callable to
            raise TypeError(
                f"the tool's argument schema cannot be inferred ({type(exc).__name__})"
            ) from None
    if isinstance(schema, dict):
        return frozenset(schema.get("properties", {}))
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return frozenset(schema.model_fields)
    raise TypeError(
        "the tool's argument schema is neither a Pydantic v2 model nor a JSON schema dict (a "
        "Pydantic v1 model is neither), so the gate cannot read the fields it declares; use "
        "a Pydantic v2 model"
    )


#: The parameter names the SDK reserves for itself (``run_manager``, ``callbacks``): it strips
#: them from every schema it generates and fills ``callbacks`` on a function and
#: ``run_manager`` on ``_run``/``_arun`` at call time, so a value under either name is never
#: the model's. ``tests/test_langgraph_gate.py`` pins the set against the SDK's own constant.
_RESERVED = frozenset(FILTERED_ARGS)


def _execution_callable(tool, is_async: bool):
    """The function the active route finally calls, chosen as the SDK chooses it.

    ``StructuredTool._run`` and ``Tool._run`` call ``func`` and their ``_arun`` calls
    ``coroutine``, falling back to ``func`` on a thread when there is none; ``BaseTool.arun``
    runs the class's ``_arun`` when it overrides the default and ``_run`` otherwise, and
    ``run`` always runs ``_run``. Bound methods are returned, so ``self`` is not among the
    parameters inspected.
    """
    if isinstance(tool, (StructuredTool, Tool)):
        target = tool.coroutine if is_async and tool.coroutine is not None else tool.func
        if target is None:
            raise TypeError(
                "the tool has no callable for this route, so the invocation cannot be judged"
            )
        return target
    if is_async and type(tool)._arun is not BaseTool._arun:
        return tool._arun
    return tool._run


def _framework_keys(func) -> frozenset[str]:
    """The parameter names the framework fills on ``func`` at call time: the reserved names
    and the parameter annotated ``RunnableConfig``, found exactly as the SDK finds it."""
    config_param = _get_runnable_config_param(func)
    return _RESERVED | ({config_param} if config_param else frozenset())


#: The keyword names ``StructuredTool`` and ``Tool`` take for themselves in ``_run`` and
#: ``_arun`` (``config``, ``run_manager``): ``BaseTool.run`` writes its own value under each
#: and the holder keeps it, so a function parameter of that name never receives the value
#: the parse step produced. ``tests/test_langgraph_gate.py`` pins the set.
_HOLDER_TAKES = frozenset(
    name
    for cls in (StructuredTool, Tool)
    for body in ("_run", "_arun")
    for name, parameter in inspect.signature(getattr(cls, body)).parameters.items()
    if parameter.kind is parameter.KEYWORD_ONLY
)


#: The one parameter name ToolNode fills whatever the parameter's annotation says: it puts
#: its runtime object under a parameter called ``runtime``. ``tests/test_compat_langgraph.py``
#: pins that against the framework.
_NODE_NAMES = frozenset({"runtime"})

#: The parameter names LangChain or ToolNode reserve: the ones they may write their own
#: value under, keep for themselves or drop from a schema, whatever the parameter's
#: annotation or default. It is the union of the gate's by-name authorities and nothing
#: else: :data:`_RESERVED` (``run_manager``, ``callbacks``), :data:`_HOLDER_TAKES`
#: (``config``, ``run_manager``) and :data:`_NODE_NAMES` (``runtime``). Used by the
#: forwarder rule only (:func:`_not_its_own`).
_RESERVED_NAMES = _RESERVED | _HOLDER_TAKES | _NODE_NAMES

#: The types that mark a parameter as one the framework or the host fills: a callback
#: manager, a run manager, and every injected-argument marker (``InjectedState``,
#: ``InjectedStore``, ``InjectedToolCallId`` and a custom one all subclass
#: ``InjectedToolArg``), each as a class, a subclass or an instance; ``ToolRuntime`` and a
#: subclass of it; ``RunnableConfig`` itself and a ``TypedDict`` that has it as a base.
_FILLED_TYPES = (BaseCallbackManager, BaseRunManager, InjectedToolArg)

#: Substrings that name a filled type, used ONLY for an annotation that is still text
#: because it could not be resolved (an unresolved forward reference). A resolved
#: annotation is never matched by text, and neither is a string that is ``Annotated``
#: metadata or a ``Literal`` value.
_FILLED_TYPE_NAMES = ("RunnableConfig", "ToolRuntime", "Injected", "CallbackManager", "RunManager")

#: PEP 695 ``type X = ...`` aliases (Python 3.12 and later).
_TYPE_ALIAS = getattr(typing, "TypeAliasType", ())

_RESOLVED, _BY_NAME = "resolved", "by name"


def _innermost(func):
    """The function at the end of ``func``'s ``functools.wraps`` chain, the one whose
    parameters the forwarder rule reads, with a ``functools.partial`` opened to the
    function it holds."""
    for _ in range(64):
        if isinstance(func, functools.partial):
            func = func.func
        elif hasattr(func, "__wrapped__"):
            func = func.__wrapped__
        else:
            break
    return func


def _hints(func) -> dict[str, Any]:
    """The innermost function's annotations as the frameworks read them: resolved with
    ``typing.get_type_hints``, so an alias under ``from __future__ import annotations``, a
    quoted name and a forward reference inside ``Optional`` are the types they name. An
    empty dict when they cannot be resolved; each raw annotation is then read on its own
    (:func:`_filled_type`)."""
    try:
        return typing.get_type_hints(_innermost(func), include_extras=True)
    except Exception:  # noqa: BLE001, unresolvable annotations are read raw, one by one
        return {}


def _namespace(func) -> dict[str, Any]:
    """The globals a forward reference in the innermost function's annotations is resolved
    against: its module's."""
    found = getattr(_innermost(func), "__globals__", None)
    return found if isinstance(found, dict) else {}


def _filled_type(annotation, namespace: dict[str, Any], _depth: int = 0) -> str | None:
    """How ``annotation`` relates to a type the framework or the host fills: ``"resolved"``
    when it is one or is built from one, ``"by name"`` when it is text that could not be
    resolved and holds the name of one, ``None`` when it does not.

    What is opened, and nothing else: text and a ``ForwardRef`` (resolved against
    ``namespace``; when that fails, matched by :data:`_FILLED_TYPE_NAMES`); a ``TypeVar``
    (its bound and constraints); a ``NewType`` (its supertype); a PEP 695 alias (its
    value); ``Annotated`` (the type and every metadata item that is not a string); a union
    and any other generic (its origin and every argument, a string among them being a
    forward reference); and a class, matched by identity, as a subclass, or through
    ``__orig_bases__`` (a ``TypedDict`` that extends ``RunnableConfig``). A ``Literal``'s
    values are not opened. An annotation nested more than 16 deep is taken for a filled
    type.
    """
    if _depth > 16:
        return _RESOLVED
    if isinstance(annotation, (str, typing.ForwardRef)):
        text = annotation if isinstance(annotation, str) else annotation.__forward_arg__
        try:
            named = eval(text, dict(namespace))  # noqa: S307, as typing.get_type_hints does
        except Exception:  # noqa: BLE001, an unresolved name is matched by its text
            return _BY_NAME if any(name in text for name in _FILLED_TYPE_NAMES) else None
        return _filled_type(named, namespace, _depth + 1)
    if annotation is RunnableConfig or isinstance(annotation, _FILLED_TYPES):
        return _RESOLVED
    if isinstance(annotation, _TYPE_ALIAS):
        inside: tuple = (annotation.__value__,)
    elif isinstance(annotation, typing.TypeVar):
        inside = (annotation.__bound__, *annotation.__constraints__)
    elif isinstance(annotation, type) and typing.get_origin(annotation) is None:
        if issubclass(annotation, (*_FILLED_TYPES, ToolRuntime)):
            return _RESOLVED
        inside = tuple(getattr(annotation, "__orig_bases__", ()))
    elif hasattr(annotation, "__supertype__"):
        inside = (annotation.__supertype__,)
    elif typing.get_origin(annotation) is typing.Literal:
        inside = ()
    elif typing.get_origin(annotation) is typing.Annotated:
        inside = tuple(item for item in typing.get_args(annotation) if not isinstance(item, str))
    else:
        inside = (typing.get_origin(annotation), *typing.get_args(annotation))
    found = None
    for inner in inside:
        if inner is None:
            continue
        how = _filled_type(inner, namespace, _depth + 1)
        if how == _RESOLVED:
            return _RESOLVED
        found = found or how
    return found


def _not_its_own(
    name: str,
    parameter,
    filled: frozenset[str],
    hints: dict[str, Any],
    namespace: dict[str, Any],
) -> str | None:
    """Why a named parameter does not count as the callable's own in the forwarder rule
    (:func:`_read_to_the_end`), and there only; None when it does count.

    It does not count when the framework or the host fills it or the tool marks it
    injected (``filled``: what :func:`_framework_keys`, :func:`_host_keys` and
    :func:`_node_fills` answer for this tool; their answer is never set aside); when its
    name is reserved (:data:`_RESERVED_NAMES`), whatever its annotation or default; or
    when its annotation, read as the frameworks read it (``hints``) and opened as
    :func:`_filled_type` opens it, is or is built from a type the framework or the host
    fills. Such a parameter says nothing about what the callable does with the model's
    arguments: ``def _run(self, *args, config: Optional[RunnableConfig] = None,
    **kwargs)`` forwards exactly as ``def _run(self, *args, **kwargs)`` does, whether or
    not LangChain fills ``config`` there. The answer is ``"by name"`` only when the one
    reason is an annotation that could not be resolved and was matched by its text.
    """
    if name in filled or name in _RESERVED_NAMES:
        return _RESOLVED
    return _filled_type(hints.get(name, parameter.annotation), namespace)


def _variadic(parameter) -> bool:
    return parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)


def _reached(func) -> list:
    """The parameters of every callable a call to ``func`` reaches, ``func``'s own first.

    ``inspect.signature`` reads a ``functools.wraps`` wrapper as the function it wraps and
    honours a declared ``__signature__``, so a wrapper's own parameter would run with its
    default while the gate looked at the wrapped function. This is the kernel's walk
    (``recusal.invocation.reached_parameters``), with every failure, of whatever type,
    turned into the ``TypeError`` the seam refuses on. One callable Python itself cannot
    read is common enough to give as the example in the reason: an object whose class
    defines ``__get__`` (``cachetools.cached`` returns one), which ``inspect.signature``
    takes for a builtin. A stack of ``functools.wraps`` decorators deeper than the walk
    follows is the other failure that has a cause the gate can state (:func:`_stacked`).

    Three refusals the kernel's walk makes by name (a kernel later than 0.15.1; beside
    0.15.0 or 0.15.1 none of the shapes is found) are reported as what they are: a decorator that
    holds a second wrapper with a parameter of its own (``HiddenWrapper``, whose text
    names the parameter and never a value), a single-dispatch function that can run more
    than its base implementation (``DispatchedAtCallTime``), and a callable that
    ``inspect.signature`` reads as a ``functools.partialmethod``
    (``PartialMethodSignature``).
    """
    try:
        return _reached_parameters(func)
    except _DeclaredSignature:
        raise TypeError(
            "the tool's callable declares a signature that is not its own (`__signature__`), "
            "so the gate cannot tell which parameters it runs with"
        ) from None
    except _PartialMethodSignature:
        raise TypeError(
            "the tool's callable is a `functools.partialmethod` over a callable that does "
            "not bind as a method (a callable object, a class-based decorator), or a "
            "wrapper of one: the signature it shows is that of what the partialmethod "
            "binds, not its own, so what runs under it cannot be read; give the tool a "
            "plain function that calls it"
        ) from None
    except _BoundArgumentKept as refused:
        raise TypeError(f"the tool's callable cannot be read: {refused}") from None
    except _UnreadCache as refused:
        raise TypeError(f"a cache on the tool's callable cannot be read: {refused}") from None
    except _HiddenWrapper as refused:
        raise TypeError(
            "a decorator on the tool's function holds a second wrapper of that function (in "
            "its closure or on the decorator object), which is not reached through "
            f"`__wrapped__` and so is not read like one applied directly: {refused}; call "
            "the decorated function from a plain function and give the tool that"
        ) from None
    except _DispatchedAtCallTime:
        raise TypeError(
            "the tool's function is a single-dispatch function (`functools.singledispatch` "
            "or `singledispatchmethod`) with an implementation registered beside its base "
            "one or in its place: the implementation that runs is chosen at call time, by "
            "the type of an argument, so its parameters and defaults cannot be read; call "
            "it from a plain function and give the tool that"
        ) from None
    except Exception as exc:  # noqa: BLE001, an uninspectable callable cannot be judged
        if _stacked(func) > _READ_DEPTH:
            raise TypeError(
                "the tool's function is under more decorators than the gate reads (more "
                f"than {_READ_DEPTH}, one inside the other), so its parameters cannot be "
                "read; call the decorated function from a plain function and give the "
                "tool that"
            ) from None
        raise TypeError(
            f"the tool's callable cannot be inspected for its parameters ({type(exc).__name__}), "
            "for example a class-based decorator that is also a descriptor (its class "
            "defines `__get__`); call the decorated function from a plain function and give "
            "the tool that"
        ) from None


#: A stack of ``functools.wraps`` decorators the kernel's walk reads to the end: it follows
#: at most this many decorators inward and raises ``ValueError`` for more. A failure on a
#: deeper stack is reported as that. ``tests/test_langgraph_gate.py`` pins the number
#: against the kernel: a stack this deep is read, one deeper is refused with this reason.
_READ_DEPTH = 31


def _stacked(func) -> int:
    """How many ``__wrapped__`` steps lead inward from ``func``, counted to one past
    :data:`_READ_DEPTH`."""
    depth = 0
    while depth <= _READ_DEPTH:
        func = getattr(func, "__wrapped__", None)
        if func is None:
            break
        depth += 1
    return depth


def _read_to_the_end(
    levels: list,
    fields: frozenset[str],
    filled: frozenset[str] = frozenset(),
    closed: bool = False,
    func: Any = None,
) -> None:
    """Raise ``TypeError`` when the innermost callable hands on an argument the gate must
    account for.

    The last callable :func:`_reached` finds is the innermost one: the walk read nothing
    further in. When it takes ``*args`` or ``**kwargs`` it can pass what it receives to code
    the walk never saw (a decorator built without ``functools.wraps``, an ``_arun`` that
    hands the call to ``_run``), and that code fills its own defaults after the policy has
    judged. So ``fields``, the names that must be accounted for, must each be a parameter
    of that callable; the variadic then carries nothing the schema declares. A callable
    with no variadic parameter cannot receive a name it does not declare, and is not
    checked here.

    A callable that takes nothing but variadics is refused whatever the schema declares.
    The schema LangChain infers from ``def wrapper(*args, **kwargs)`` declares ``args`` and
    ``kwargs``, the variadics' own names, so every field would count as a parameter while
    the wrapper hands everything to a function whose defaults nobody read. A variadic's
    own name accounts for a field only on a callable that also names a parameter
    (``def f(q, **kwargs)``).

    A parameter the framework or the host fills, one with a reserved name, or one
    annotated with a type the framework or the host fills is not a parameter of the
    callable's own for this purpose (:func:`_not_its_own`; ``func`` is the callable the
    annotations are resolved for): ``def wrapper(*args, config: RunnableConfig, **kwargs)`` forwards
    exactly as ``def wrapper(*args, **kwargs)`` does, and so does the same wrapper with
    ``config: Optional[RunnableConfig] = None`` or a bare ``config=None``. The one such
    callable that is let through is one under a schema the model can put nothing through
    (``closed``, :func:`_closed_to_the_model`): the call then carries no value the model
    chose, and what the function does with nothing is its own body.
    """
    innermost = levels[-1]
    if not any(_variadic(parameter) for parameter in innermost.values()):
        return
    unread = sorted(fields - innermost.keys())
    hints, namespace = _hints(func), _namespace(func)
    why = {
        name: _not_its_own(name, parameter, filled, hints, namespace)
        for name, parameter in innermost.items()
        if not _variadic(parameter)
    }
    forwards_everything = all(why.values())
    if not unread and forwards_everything and not closed:
        kept = ", ".join(f"`{name}`" for name, how in why.items() if how == _RESOLVED)
        unresolved = ", ".join(f"`{name}`" for name, how in why.items() if how == _BY_NAME)
        besides = f" (and {kept}, which LangChain or the host reserves)" if kept else ""
        if unresolved:
            besides += (
                f" (and {unresolved}, whose type could not be resolved and was matched by "
                "name as one LangChain or the host reserves)"
            )
        raise TypeError(
            f"the tool's innermost callable takes only `*args` or `**kwargs`{besides}, so it "
            "passes every argument on to code the gate cannot read; name the arguments in "
            "its signature, build a decorator with `functools.wraps`, or give the tool the "
            "function itself"
        )
    if unread:
        raise TypeError(
            "the tool's innermost callable takes `*args` or `**kwargs` and does not name "
            f"`{unread[0]}`, so it passes that argument on to code the gate cannot read; "
            "name every argument in its signature, build a decorator with `functools.wraps`, "
            "or give the tool the function itself"
        )


def _closed_to_the_model(tool) -> bool:
    """Whether the tool's schema lets the model put nothing through to the function.

    True only for a Pydantic v2 ``args_schema`` that declares no field and does not set
    ``extra="allow"``: LangChain validates the input against it and hands the function
    only the fields it declares, so every key the model sends is dropped. Everything else
    lets a value through and is not closed: a schema with a field (the schema inferred
    from ``def wrapper(*args, **kwargs)`` declares ``args`` and ``kwargs``); a JSON schema
    dict, whatever its ``properties`` and ``additionalProperties`` say, because LangChain
    validates nothing against a dict and passes every key on; and no schema at all on a
    ``BaseTool`` subclass, for the same reason.
    """
    schema = tool.args_schema
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        return False
    return not schema.model_fields and schema.model_config.get("extra") != "allow"


#: The names ``run_in_executor`` takes for itself. A ``BaseTool`` subclass with no ``_arun``
#: of its own runs ``_run`` through it on the async routes (``BaseTool._arun``), with the
#: arguments as keywords, so a key of one of these names collides with the runner's own
#: parameter. Read from the function's signature; ``tests/test_compat_langgraph.py`` pins
#: that ``BaseTool._arun`` still runs the tool this way.
_THREAD_TAKES = frozenset(
    name
    for name, parameter in inspect.signature(run_in_executor).parameters.items()
    if not _variadic(parameter)
)


def _runs_on_a_thread(tool, is_async: bool) -> bool:
    """Whether this route hands the tool's arguments, as keywords, to ``run_in_executor``:
    the async route of a ``BaseTool`` subclass that has no ``_arun`` of its own. A
    ``StructuredTool`` or ``Tool`` without a coroutine does not: its ``ainvoke`` runs the
    whole sync ``invoke`` on a thread, with the arguments still inside the input, and its
    class has an ``_arun`` of its own."""
    return is_async and type(tool)._arun is BaseTool._arun


#: Where ``langchain-mcp-adapters`` defines the function every tool it builds runs: the
#: module, the function that builds a tool, and the closure variable that holds the tool's
#: interceptors. ``tests/test_langgraph_gate.py`` pins the three against the library when it
#: is installed.
_MCP_MODULE = "langchain_mcp_adapters.tools"
_MCP_BUILDER = "convert_mcp_tool_to_langchain_tool"
_MCP_INTERCEPTORS = "tool_interceptors"


def _calls_an_mcp_server(func) -> bool:
    """Whether ``func`` is the function ``langchain-mcp-adapters`` defines to call an MCP
    server, built with no tool interceptor.

    That function is ``call_tool(runtime=None, **arguments)`` under the server's JSON
    schema: it names none of the schema's fields, which :func:`_read_to_the_end` refuses.
    But what it hands on is the judged arguments as they stand, to the server and not to
    Python code with defaults of its own, so it is the one exception to that rule.

    It is recognised by identity, not by its looks: its code object must be the very one
    the library's builder function holds among its constants, in the module as it is
    imported in this process. A function of the same name and signature from anywhere else
    is not it, and neither is a wrapper around it (the callable the tool runs must be the
    function itself). The model can influence none of this.

    A tool interceptor (``tool_interceptors=``) runs between that function and the server
    and can replace the arguments, after the policy has judged them. So a tool built with
    one raises ``TypeError`` and is refused; and so is one from a release whose function
    does not hold its interceptors where this reads them (before 0.1.12 there were none,
    and the function did not have the variable), because the gate could not tell.
    """
    builder = getattr(sys.modules.get(_MCP_MODULE), _MCP_BUILDER, None)
    if not (inspect.isfunction(func) and inspect.isfunction(builder)):
        return False
    code = func.__code__
    if not any(constant is code for constant in builder.__code__.co_consts):
        return False
    try:
        cells = func.__closure__ or ()
        held = cells[code.co_freevars.index(_MCP_INTERCEPTORS)].cell_contents
    except (ValueError, TypeError, IndexError):
        raise TypeError(
            "the tool was built by langchain-mcp-adapters, and the installed release does "
            "not show the gate whether the tool has interceptors that can change its "
            "arguments on the way to the server; use langchain-mcp-adapters 0.1.12 or later"
        ) from None
    if held:
        raise TypeError(
            "the tool was built by langchain-mcp-adapters with tool interceptors, which can "
            "change the arguments on the way to the server, after the policy has judged "
            "them; build the tool without interceptors, or give the node a function of "
            "your own that names the parameters and calls the tool"
        )
    return True


def _fills_itself(tool, func) -> frozenset[str]:
    """The keyword names under which ``func`` would not receive a value the parse step
    produced, because LangChain writes its own value there or keeps the name for itself.

    For the function a ``StructuredTool`` or ``Tool`` holds: ``config`` and ``run_manager``
    always (``BaseTool.run`` writes both, and the holder's ``_run`` takes them as its own
    keywords: :data:`_HOLDER_TAKES`), the function's ``RunnableConfig`` parameter, and
    ``callbacks`` when the function names a parameter ``callbacks``, which the holder then
    overwrites. For a ``BaseTool`` subclass's ``_run``/``_arun``: ``run_manager`` when the
    method names it, and its ``RunnableConfig`` parameter; nothing writes ``callbacks``
    there. A ``callbacks`` key that reaches a function through ``**kwargs`` is not written
    by anything either: the function receives it as judged, so it is judged like any other.

    The signature is read as LangChain reads it (``inspect.signature``, which follows
    ``__wrapped__``: a ``callbacks`` parameter of the function behind a ``functools.wraps``
    decorator counts). If it cannot be read, for whatever reason (Python 3.14 raises
    ``NameError`` for an annotation it cannot resolve), every reserved name counts as
    filled: refused, not guessed.
    """
    config_param = _get_runnable_config_param(func)
    names = {config_param} if config_param else set()
    try:
        parameters = inspect.signature(func).parameters
    except Exception:  # noqa: BLE001, a signature that cannot be read is not guessed at
        return frozenset(names | _RESERVED | _HOLDER_TAKES)
    if isinstance(tool, (StructuredTool, Tool)):
        names |= _HOLDER_TAKES
        if "callbacks" in parameters:
            names.add("callbacks")
    elif "run_manager" in parameters:
        names.add("run_manager")
    return frozenset(names)


def _completed(
    kwargs: dict[str, Any], tool, host_keys: frozenset[str], is_async: bool
) -> dict[str, Any]:
    """The parsed kwargs, held to every callable the route reaches, with every suppressed
    schema parameter restored from the default that would run.

    Python fills whatever the parsed kwargs leave out from a callable's own defaults, after
    the policy has judged, so the callables are the authority on what will execute: the one
    the route calls and every one it hands the call to (:func:`_reached`: a
    ``functools.wraps`` wrapper and the function it wraps, a method, a ``functools.partial``,
    a callable object). Each named parameter one of them declares must be a field the schema
    declares (judged), a key the host filled for this call (never the model's) or a name the
    framework fills at call time (:func:`_framework_keys`); any other parameter is a value
    the proposal cannot carry and the policy would never see, so the call is refused as
    unjudgeable, naming the parameter and never its default. A positional-only parameter
    cannot be judged by name and is refused too. So is a parameter that a forwarding wrapper
    (``*args``/``**kwargs``) names and another callable on the way names as well: the wrapper
    keeps the judged argument and the other runs with its own default. And the innermost
    callable must not hand a schema field on to code the walk could not read
    (:func:`_read_to_the_end`), unless the callable is the one that hands its arguments to
    an MCP server as they stand (:func:`_calls_an_mcp_server`).

    ``_parse_input`` returns only the fields present in the validated model's
    ``model_dump()``, so a schema field with ``exclude=True`` or a model-wide serializer that
    omits it never reaches the kwargs, and the function would fill its own Python default
    after the policy judged an invocation without it; a tool with no declared schema is not
    validated at all, so whatever the model left out is missing too. Each parameter the
    schema declares, the host did not fill, and the parse did not return is taken from the
    default of the first callable that names it, and passed explicitly, so the judged
    invocation is the executed one; a missing parameter with no default raises
    ``TypeError`` and the call is refused as unjudgeable.

    Last, a judged argument under a name LangChain fills itself (:func:`_fills_itself`:
    ``config`` and ``run_manager`` on a ``StructuredTool``, ``callbacks`` on a function that
    names it) is refused: the framework would overwrite or keep it, and the function would
    not receive that value. So is one named ``self``: LangChain calls the tool's bound
    ``_run``/``_arun`` with the arguments as keywords, and Python raises for a second
    ``self`` after the policy has allowed the call. And so is one named like a parameter of
    ``run_in_executor`` on a route that runs the tool through it (:data:`_THREAD_TAKES`),
    for the same reason.
    """
    fields = _schema_fields(tool)
    func = _execution_callable(tool, is_async)
    levels = _reached(func)
    to_a_server = _calls_an_mcp_server(func)
    framework_keys = _framework_keys(func)
    if not to_a_server:
        filled = framework_keys | _host_keys(tool) | _node_fills(tool)
        _read_to_the_end(levels, fields, filled, _closed_to_the_model(tool), func)
    named = [{n for n, p in level.items() if not _variadic(p)} for level in levels]
    completed = dict(kwargs)
    for index, level in enumerate(levels):
        forwards = any(_variadic(parameter) for parameter in level.values())
        for name, parameter in level.items():
            if _variadic(parameter):
                continue
            if name in host_keys or name in framework_keys:
                continue
            if parameter.kind is parameter.POSITIONAL_ONLY:
                raise TypeError(
                    f"the callable's parameter `{name}` is positional-only, so the invocation "
                    "cannot be judged by name"
                )
            if name not in fields:
                raise TypeError(
                    f"the callable's parameter `{name}` is absent from the argument schema, so "
                    "the invocation cannot be judged; declare it in the schema"
                )
            if forwards and any(name in other for i, other in enumerate(named) if i != index):
                raise TypeError(
                    f"the callable's parameter `{name}` is taken by a wrapper and by the "
                    "function it wraps, which would then run with its own default, a value "
                    "no policy judged"
                )
            if name in completed:
                continue
            if parameter.default is parameter.empty:
                if tool.args_schema is None:
                    raise TypeError(
                        f"the call leaves out `{name}` and the callable has no default for "
                        "it, so the invocation cannot be judged"
                    )
                raise TypeError(
                    f"the schema suppressed `{name}` and the callable has no default for it, "
                    "so the invocation cannot be judged"
                )
            completed[name] = parameter.default
    if "self" in completed and "self" not in host_keys:
        raise TypeError(
            "the invocation sets `self`, which is the tool object itself when LangChain "
            "calls the tool, so the call could not be made; an argument named `self` "
            "cannot be passed to a tool"
        )
    if _runs_on_a_thread(tool, is_async):
        clashing = sorted(k for k in completed if k in _THREAD_TAKES)
        own = ", ".join(f"`{k}`" for k in clashing if any(k in level for level in levels))
        sent = ", ".join(f"`{k}`" for k in clashing if not any(k in level for level in levels))
        if own:
            raise TypeError(
                f"the tool's own parameter {own} has the name of a parameter of the "
                "function LangChain runs this tool with on a thread (the tool has no `_arun` "
                "of its own), so the judged value cannot be passed to it on the async "
                "routes; rename the parameter or define `_arun`"
            )
        if sent:
            raise TypeError(
                f"the invocation sets {sent}, a parameter of the function LangChain runs "
                "this tool with on a thread (the tool has no `_arun` of its own), so "
                "the call could not be made; an argument of that name cannot be passed to "
                "this tool on the async routes"
            )
    fills = _fills_itself(tool, func)
    taken = ", ".join(f"`{k}`" for k in sorted(completed) if k not in host_keys and k in fills)
    if taken and to_a_server:
        raise TypeError(
            f"the invocation sets {taken}, which LangChain fills itself at call time, so "
            "the MCP server would not receive it; a server argument with that name cannot "
            "be sent through langchain-mcp-adapters"
        )
    if taken:
        raise TypeError(
            f"the invocation sets {taken}, which LangChain fills itself at call time, so "
            "the tool would not receive the judged value; rename the parameter"
        )
    return completed


def _single_input(tool, args: tuple, kwargs: dict[str, Any], is_async: bool) -> str:
    """The name of the parameter a single-input ``Tool`` hands its one value to.

    A ``Tool`` (``Tool(name, func, description)``, ``@tool(infer_schema=False)``) calls its
    function with one positional value and no names, whatever key the model wrote it under.
    That value can be judged by name only when the name is certain: the function's own first
    parameter takes it, and no callable the call reaches names any other parameter (one with
    a default would run with a value no policy judged), and the innermost callable, if it
    takes ``*args`` or ``**kwargs``, names that parameter too (:func:`_read_to_the_end`).
    Anything else, and a positional input to any other kind of tool, raises ``TypeError``
    and the call is refused.
    """
    if not isinstance(tool, Tool) or kwargs or len(args) != 1:
        raise TypeError("a positional tool input cannot be judged by name")
    func = _execution_callable(tool, is_async)
    levels = _reached(func)
    framework_keys = _framework_keys(func)
    first = next(iter(levels[0].items()), None)
    if (
        first is None
        or first[1].kind is not first[1].POSITIONAL_OR_KEYWORD
        or first[0] in framework_keys
    ):
        raise TypeError(
            "the single-input tool's function does not take its input as a first named "
            "parameter, so the input cannot be judged by name"
        )
    for level in levels:
        for name, parameter in level.items():
            if _variadic(parameter) or name in framework_keys or name == first[0]:
                continue
            raise TypeError(
                f"the callable's parameter `{name}` is not the tool's single input, so it "
                "would run with a value no policy judged"
            )
    _read_to_the_end(levels, frozenset({first[0]}), func=func)
    return first[0]


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


#: Error types whose message is written by user code (a validator's ``raise ValueError``
#: or ``assert``), so it may echo anything the validator could see, including host state.
_USER_WRITTEN = frozenset({"value_error", "assertion_error"})


def _sanitized(exc: Exception, host_keys: frozenset[str]) -> str:
    """A validation failure reduced to what is demonstrably safe to show the model and to
    put on the record.

    Pydantic's ``errors(include_input=False, include_context=False)`` already leaves the
    input out, but a message is still forwarded only when it is one of Pydantic's own
    templates for a model-facing field: an error at the model root (a ``model_validator``,
    location ``()``), an error under a host-supplied argument (whose value may be graph
    state), and an error of a user-written type (``value_error``, ``assertion_error``, whose
    text a validator composed and may have built from host context) are reported as
    location and error type only. A non-Pydantic exception is its class name."""
    if isinstance(exc, ValidationError):
        parts = []
        for error in exc.errors(include_url=False, include_context=False, include_input=False):
            loc = tuple(str(part) for part in error.get("loc", ()))
            kind = str(error.get("type"))
            where = ".".join(loc) or "model"
            if not loc or loc[0] in host_keys or kind in _USER_WRITTEN:
                parts.append(f"{where}: {kind}")
            else:
                parts.append(f"{where}: {error.get('msg')}")
        return "; ".join(parts)
    return type(exc).__name__


def _unchanged(arrived: Any, sent: Any) -> bool:
    """Whether the value that reached the tool is the one the model sent."""
    if arrived is sent:
        return True
    try:
        return bool(arrived == sent)
    except Exception:  # noqa: BLE001, a value that cannot be compared was not shown to be replaced
        return True


def _unjudgeable(name: str, detail: str, bound: dict, policy: Policy, record: dict) -> str:
    """The reason for a call that cannot be judged, recorded through ``decide`` like every
    other refusal (so an audited gate logs it)."""

    def unjudgeable(_name, _input):
        return [Finding.fail("recusal_unjudgeable_input", severity="CRITICAL", message=detail)]

    try:
        _, reason = decide(name, bound, unjudgeable, safe_errors=True, **record)
    except Exception as exc:  # noqa: BLE001, the refusal stands even if the record cannot be written
        reason = (
            f"Recusal failed closed (policy error): RECUSAL_POLICY_ERROR ({type(exc).__name__})"
        )
    return reason


def _judging(
    tool,
    call: dict[str, Any],
    policy: Policy,
    record: dict[str, Any],
    is_async: bool,
    entered: list | None = None,
):
    """A per-call copy of ``tool`` that judges, then runs, exactly the invocation it parsed.

    ``BaseTool.run``/``arun`` parse the input through ``_to_args_and_kwargs`` and hand the
    result to ``_run``/``_arun``; ToolNode injects the host's state, store and runtime into
    the input before that (``_inject_tool_args``), whatever hook or middleware ran earlier.
    The copy's parse step therefore sees the final input: it runs the tool's own parser once,
    holds the callable this route executes (``is_async`` names the route) to the schema and
    completes any schema parameter that parse suppressed from that callable's default
    (:func:`_completed`), projects every model-controlled argument with :func:`_faithful`,
    calls ``decide`` once, and on allow returns the very objects it judged, so the tool
    executes what was judged with no second validation and nothing to compare. A
    single-input ``Tool`` hands its function one positional value; it is judged under the
    name of the parameter that takes it (:func:`_single_input`). A deny, an input the parser
    rejects, any other positional input, or a value that cannot be projected raises
    :class:`RecusalRefusal`, which the copy's ``handle_tool_error`` turns into the error
    ToolMessage the model reads; any other ``ToolException`` keeps the original tool's
    handling. The copy serves one call id; every execution attempt of that call (a retry, a
    resume) is judged again.

    ``entered`` is a list the parse step appends to each time it is entered: ``True`` once
    it has allowed that execution, and otherwise the refusal it answered with. The hook that
    passes one can tell, once execution returns, whether this call passed the judging step
    at all and what the step last did (:func:`_judge`).
    """
    name, call_id, fallback = call["name"], call.get("id"), call.get("args")
    copy = tool.model_copy()
    #: Holds the arguments the parse step just judged and allowed, positional and keyword,
    #: from that moment until the tool body takes them. The body runs only by spending this,
    #: and only with those very objects.
    allowed: list[tuple[tuple, dict[str, Any]]] = []
    entered = [] if entered is None else entered
    parse = type(tool)._to_args_and_kwargs
    host_keys = _host_keys(tool)
    node_fills = _node_fills(tool)
    original_handler = tool.handle_tool_error

    def proposal(tool_input) -> dict:
        """The model-facing input, for the record of a refusal that judged nothing."""
        if isinstance(tool_input, dict):
            try:
                return _faithful({k: v for k, v in tool_input.items() if k not in host_keys})
            except TypeError:
                pass
        return fallback if isinstance(fallback, dict) else {"input": fallback}

    def from_the_host(tool_input) -> frozenset[str]:
        """The host keys the host really filled for this call.

        A key counts as the host's only because ToolNode strips it from the model's
        arguments and puts its own value there, which it does for state, store and runtime
        and for nothing else (:func:`_node_fills`). A value that arrives under any other
        injected name was not stripped, so it is not the host's: a function can mark a
        parameter injected that the schema declares as an ordinary one, ToolNode reads the
        schema, and the value is then the model's, or the one a hook or middleware after
        the gate wrote over it. The tool's annotations say which keys ToolNode should fill,
        and they can disagree with what it does. So they are not trusted alone either: a
        key the model sent whose value arrives here unchanged was not replaced by the
        host. It is the model's, and it is judged like every other argument.

        A key that is absent from the input was not filled by the host either. The parse
        step fills it afterwards: an ``InjectedToolCallId`` from the call id, which the model
        writes, and any other from its schema default. Both are values the function runs
        with that no host supplied, so they are judged. This is read before the parse step,
        which writes the call id into the input it is given.
        """
        if not isinstance(tool_input, dict):
            return frozenset()
        filled = node_fills & tool_input.keys()
        if not isinstance(fallback, dict):
            return filled
        return filled - {
            key for key in filled & fallback.keys() if _unchanged(tool_input[key], fallback[key])
        }

    def seam(tool_input, tool_call_id):
        if tool_call_id != call_id:
            raise RuntimeError("Recusal: a judging copy of a tool was used for a different call")
        entered.append(None)
        try:
            judged = judge(tool_input, tool_call_id)
        except RecusalRefusal as refusal:
            entered[-1] = str(refusal)
            raise
        entered[-1] = True
        return judged

    def judge(tool_input, tool_call_id):
        hosts = from_the_host(tool_input)
        try:
            args, kwargs = parse(copy, tool_input, tool_call_id)
        except Exception as exc:  # noqa: BLE001, an input the tool rejects is refused with its reason
            detail = f"arguments do not validate ({_sanitized(exc, host_keys)})"
            raise RecusalRefusal(_unjudgeable(name, detail, proposal(tool_input), policy, record))
        try:
            if args:
                judged = {_single_input(tool, args, kwargs, is_async): _faithful(args[0])}
            else:
                kwargs = _completed(kwargs, tool, hosts, is_async)
                judged = {k: _faithful(v) for k, v in kwargs.items() if k not in hosts}
        except Exception as exc:  # noqa: BLE001, an invocation that cannot be prepared is refused
            # a TypeError is one of the gate's own reasons; anything else is reported by
            # its class alone, since its text is not the gate's
            detail = (
                str(exc)
                if isinstance(exc, TypeError)
                else f"the invocation could not be prepared for judgment ({type(exc).__name__})"
            )
            raise RecusalRefusal(_unjudgeable(name, detail, proposal(tool_input), policy, record))
        try:
            decision, reason = decide(name, judged, policy, safe_errors=True, **record)
        except Exception as exc:  # noqa: BLE001, a buggy policy must not silently disable the gate
            decision, reason = (
                "deny",
                f"Recusal failed closed (policy error): RECUSAL_POLICY_ERROR ({type(exc).__name__})",
            )
        if decision == "deny":
            raise RecusalRefusal(reason)
        # a copy: the framework adds its own call-time names to the dict it is handed back
        allowed[:] = [(tuple(args), dict(kwargs))]
        return args, kwargs

    def spend(func, args, kwargs):
        """Refuse a tool body unless it is entered with the invocation the parse step just
        judged and allowed: the same objects in the same positions and under the same names,
        and beyond them only the names the framework fills at call time."""
        if not allowed:
            detail = (
                "the tool was reached without passing the gate's judging step, so the "
                "invocation was never judged"
            )
            raise RecusalRefusal(_unjudgeable(name, detail, proposal(fallback), policy, record))
        positional, judged = allowed.pop()
        same = (
            len(args) == len(positional)
            and all(arrived is sent for arrived, sent in zip(args, positional))
            and all(k in kwargs and kwargs[k] is v for k, v in judged.items())
        )
        if not same or not set(kwargs) - set(judged) <= _framework_keys(func):
            detail = (
                "the tool was reached with arguments other than the ones the gate judged, "
                "so this invocation was never judged"
            )
            raise RecusalRefusal(_unjudgeable(name, detail, proposal(fallback), policy, record))

    def guarded(func):
        if func is None:
            return None
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def abody(*args, **kwargs):
                spend(func, args, kwargs)
                return await func(*args, **kwargs)

            return abody

        @functools.wraps(func)
        def body(*args, **kwargs):
            spend(func, args, kwargs)
            return func(*args, **kwargs)

        return body

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
    if isinstance(tool, (StructuredTool, Tool)):
        object.__setattr__(copy, "func", guarded(tool.func))
        object.__setattr__(copy, "coroutine", guarded(tool.coroutine))
    else:
        object.__setattr__(copy, "_run", guarded(tool._run))
        if type(tool)._arun is not BaseTool._arun:
            object.__setattr__(copy, "_arun", guarded(tool._arun))
    return copy


_NEVER_JUDGED = (
    "the call was answered without passing the gate's judging step, so it was never judged: "
    "the installed LangGraph, or a hook or middleware between the gate and the tool, ran or "
    "answered it another way. If the tool ran, what it did has happened; its result is "
    "withheld"
)

_NOT_THE_REFUSAL = (
    "the gate's judging step did not allow the call, and the answer that came back is not "
    "the gate's refusal: a hook or middleware between the gate and the tool ran another "
    "tool or answered in its place. If a tool ran, what it did has happened; its result is "
    "withheld"
)

_ANOTHER_TOOL = (
    "the call was answered under the name of another tool than the one the gate judged: a "
    "hook or middleware between the gate and the tool ran another tool or answered in its "
    "place, or the tool gave its own answer another name. If a tool ran, what it did has "
    "happened; its result is withheld"
)


def _settled(result, entered: list, name: str) -> str | None:
    """Why the executor's answer cannot be passed on as the answer to the call the judging
    step handled, or None when it can.

    ``entered`` is what the copy's parse step recorded, one item per execution it was
    entered for (:func:`_judging`). Nothing there: the call never passed the step. The last
    item not ``True``: the step refused the last execution it saw, so the only answer it
    produced is that refusal, an error ``ToolMessage`` with the reason as its content. An
    error ``ToolMessage`` whose text still holds that reason is taken for it (LangChain's
    human-in-the-loop middleware puts a note in front of the answer to a call a reviewer
    edited); anything else came from somewhere the gate did not judge. The last item
    ``True``: the step allowed that execution, and a ``ToolMessage`` that names another
    tool is not taken for its answer (a tool that returns its own ``ToolMessage`` may leave
    the name out, or use the tool's).
    """
    if not entered:
        return _NEVER_JUDGED
    allowed = entered[-1]
    if allowed is not True:
        own = (
            isinstance(result, ToolMessage)
            and result.status == "error"
            and isinstance(allowed, str)
            and isinstance(result.content, str)
            and allowed in result.content
        )
        return None if own else _NOT_THE_REFUSAL
    if isinstance(result, ToolMessage) and result.name not in (None, name):
        return _ANOTHER_TOOL
    return None


def _judge(request, policy: Policy, *, is_async: bool = False, **record: Any):
    """Return ``(refusal, request_to_execute, settle)``: a refusal ToolMessage for a call
    that can never reach the seam, else the request with its tool replaced by the judging
    copy, and the check the hook passes the executor's answer through.

    Refused here, before the policy runs, each through ``decide`` so an audited gate records
    it: a call whose ``id`` is empty or not a string (the entry is written without a call
    id, which the kernel does not accept empty); a call for a tool this node does not have;
    a call whose name is not the name of the tool that will run (an earlier hook attached
    another tool: the policy would judge, and the record would name, something that does not
    run; the refusal is made under the tool's own registered name); a tool that overrides
    the standard lifecycle; and a tool whose host-injected arguments cannot be read.

    ``settle(result)`` returns ``result`` when it fits what the judging copy's parse step
    last did for this call, and otherwise a refusal in its place (:func:`_settled`): the
    step never ran (an installed LangGraph that runs the tool another way, a hook or
    middleware after the gate that swaps the tool or answers for it), or it refused and the
    answer is something other than that refusal (a hook or middleware after the gate that
    answers a refused call by running another tool), or the answer names another tool. A
    tool may already have run by then; the refusal says so, and is one more entry on an
    audited gate's log.

    ``is_async`` names the route the hook runs on (``awrap_tool_call`` executes the tool
    through ``arun``), so the copy holds the callable that route runs. ``record`` goes
    straight to ``decide`` (``audit=``, ``surface=``, ``tool_use_id=``, ...), so a gate that
    puts the decision on a log (examples/langgraph_audit.py) decides, and refuses, exactly as
    this one does: one function, no second copy to drift.
    """
    call = request.tool_call
    name, call_id, raw = call["name"], call.get("id"), call.get("args")
    tool = request.tool
    has_id = isinstance(call_id, str) and bool(call_id)
    if not has_id and "tool_use_id" in record:
        record = {**record, "tool_use_id": None}
    # the name every refusal is made and recorded under: the tool's own when there is one
    known = name if tool is None else tool.name

    def refuse(detail):
        bound = raw if isinstance(raw, dict) else {"input": raw}
        reason = _unjudgeable(known, detail, bound, policy, record)
        answered = call_id if isinstance(call_id, str) else ""
        return ToolMessage(content=reason, tool_call_id=answered, name=known, status="error")

    if tool is None:
        return refuse("no tool by that name is registered with this node"), None, None
    if name != tool.name:
        return (
            refuse(
                "the call names another tool than this one, which is the tool that would "
                "run, so the name the policy judges would not be the tool that runs"
            ),
            None,
            None,
        )
    if not has_id:
        return (
            refuse(
                "the tool call has no id (it is empty or not a string), so its answer and "
                "its record cannot be tied to the call"
            ),
            None,
            None,
        )
    why = _lifecycle_refusal(tool)
    if why is not None:
        return (
            refuse(
                f"the tool's execution path cannot be judged ({why}); the gate judges inside "
                "the standard BaseTool lifecycle, so build the tool with @tool or StructuredTool"
            ),
            None,
            None,
        )
    entered: list[bool] = []
    try:
        judging = _judging(tool, call, policy, record, is_async, entered)
    except TypeError as exc:
        return refuse(str(exc)), None, None

    def settle(result):
        why = _settled(result, entered, tool.name)
        return result if why is None else refuse(why)

    return None, request.override(tool=judging), settle


def _gated(request, execute, policy: Policy, **record: Any):
    """One sync hook call: refuse, or execute the judging copy and settle its answer."""
    refusal, judged, settle = _judge(request, policy, **record)
    return refusal if refusal is not None else settle(execute(judged))


async def _agated(request, execute, policy: Policy, **record: Any):
    """One async hook call, the same decision on the async route."""
    refusal, judged, settle = _judge(request, policy, is_async=True, **record)
    return refusal if refusal is not None else settle(await execute(judged))


#: Every hook this package has built, by ``id``: a weak reference to it, the ToolNode slot
#: it belongs in, and the policy and audit log it was built for. ``is_gated`` recognises a
#: hook by being the very object in here, which an attribute copied onto another function
#: (``functools.wraps`` copies them) or an object that compares equal cannot imitate.
_HOOKS: dict[int, tuple[Any, str, Any, Any]] = {}

_SYNC_IN_ASYNC_SLOT = (
    "{name} is in a ToolNode's async slot (awrap_tool_call). That leaves the sync slot "
    "empty, and a ToolNode's sync route (invoke, stream) then runs every tool with no hook. "
    "Attach both hooks with gated_tool_node(tools, policy) or "
    "ToolNode(tools, **recusal_gates(policy)), or this one with wrap_tool_call."
)

_ASYNC_ALONE = (
    "{name} on its own gates only the async route: a ToolNode given only awrap_tool_call "
    "runs every tool with no hook on invoke and stream. Attach both hooks with "
    "gated_tool_node(tools, policy) or ToolNode(tools, **recusal_gates(policy)). Pass "
    "host_refuses_sync_runs=True only for a host that refuses a sync run when it is given "
    "an async hook, as LangChain's create_agent middleware does; if this came from "
    "recusal-langchain, upgrade it to 0.1.2 or later."
)


def _marked(hook, slot: str, policy: Any, log: Any = None):
    key = id(hook)
    _HOOKS[key] = (weakref.ref(hook, lambda _: _HOOKS.pop(key, None)), slot, policy, log)
    return hook


def _require_sync_slot(name: str, execute) -> None:
    """Raise if a sync hook is running from the async slot. ToolNode hands the async slot
    an ``async def`` executor and the sync slot a plain one, on every route."""
    if inspect.iscoroutinefunction(execute):
        raise TypeError(_SYNC_IN_ASYNC_SLOT.format(name=name))


def _require_sync_refusal(name: str, host_refuses_sync_runs: bool) -> None:
    if host_refuses_sync_runs is not True:
        raise TypeError(_ASYNC_ALONE.format(name=name))


def recusal_gate(policy: Policy):
    """Return a synchronous ToolNode hook that gates each call before execution.

    Attached alone (``wrap_tool_call=``) it covers every route: a ToolNode with no async
    hook runs this one on ``ainvoke`` too. A tool that has only a coroutine is then refused
    there, since this hook judges the sync callable; ``recusal_gates`` attaches both.

    It belongs in the sync slot only. In the async slot (``awrap_tool_call=``) it would
    leave the sync slot empty, so there it raises ``TypeError`` instead of gating one route
    and leaving the other open.
    """

    def gate(request, execute):
        _require_sync_slot("recusal_gate", execute)
        return _gated(request, execute, policy)

    return _marked(gate, "sync", policy)


def _agate(policy: Policy):
    async def gate(request, execute):
        return await _agated(request, execute, policy)

    return _marked(gate, "async", policy)


def recusal_agate(policy: Policy, *, host_refuses_sync_runs: bool = False):
    """Return the asynchronous hook on its own, for a host that refuses a sync run itself.

    A ``ToolNode`` is not such a host: given only ``awrap_tool_call`` it runs every tool
    with no hook on ``invoke`` and ``stream``. So this raises ``TypeError`` unless the
    caller states ``host_refuses_sync_runs=True``. For a ToolNode use ``recusal_gates`` or
    ``gated_tool_node``.
    """
    _require_sync_refusal("recusal_agate", host_refuses_sync_runs)
    return _agate(policy)


def recusal_gates(policy: Policy) -> dict[str, Any]:
    """Both hooks, keyed by the ToolNode parameter each one fills.

        node = ToolNode(tools, **recusal_gates(policy))

    The sync route runs the sync hook and the async route the async one, with the same
    decision logic, so every way of running the node crosses the gate.
    """
    return {"wrap_tool_call": recusal_gate(policy), "awrap_tool_call": _agate(policy)}


def _no_hooks_among(options: dict[str, Any]) -> None:
    for name in ("wrap_tool_call", "awrap_tool_call"):
        if name in options:
            raise TypeError(
                f"{name} is the gate's own slot; a ToolNode has one of each, so another "
                "hook there would replace the gate on that route"
            )


def _checked(node: ToolNode) -> ToolNode:
    """The node, or ``TypeError`` for a tool the gate would refuse on every call for a
    reason its definition already shows: a ``StructuredTool`` or ``Tool`` whose function
    names a parameter ``config`` that is not its ``RunnableConfig``. LangChain keeps that
    name for the run configuration, so the function never receives the model's value
    (:func:`_fills_itself`). The first parameter of a single-input ``Tool``'s function is
    not such a parameter: it is handed the tool's one value by position, whatever it is
    called, and the call is judged under that name. A function that cannot be inspected
    here is left to the gate, which refuses it when it is called."""
    for tool in node.tools_by_name.values():
        if not isinstance(tool, (StructuredTool, Tool)):
            continue
        for func in (tool.func, tool.coroutine):
            if func is None:
                continue
            try:
                levels, framework_keys = _reached(func), _framework_keys(func)
            except Exception:  # noqa: BLE001, refused at call time, with the reason
                continue
            single_input = next(iter(levels[0]), None) if isinstance(tool, Tool) else None
            for level in levels:
                for name, parameter in level.items():
                    if _variadic(parameter) or name in framework_keys:
                        continue
                    if name in _HOLDER_TAKES and name != single_input:
                        raise TypeError(
                            f"tool `{tool.name}`: its function's parameter `{name}` has a "
                            "name LangChain fills itself at call time, so the function would "
                            "never receive the model's value and the gate would refuse every "
                            "call; rename the parameter"
                        )
    return node


def gated_tool_node(tools, policy: Policy, **options: Any) -> ToolNode:
    """A ``ToolNode`` over ``tools`` with the gate on both of its hook slots.

    ``options`` are ToolNode's own (``name``, ``tags``, ``handle_tool_errors``,
    ``messages_key``). Passing either hook slot raises ``TypeError``. So does a tool whose
    function names a parameter ``config`` that is not its ``RunnableConfig``
    (:func:`_checked`).
    """
    _no_hooks_among(options)
    return _checked(ToolNode(tools, **recusal_gates(policy), **options))


def is_gated(node: Any) -> bool:
    """True only for a ``ToolNode`` this function can vouch for on every route.

    It requires all of:

    - the node is exactly a ``ToolNode``, not a subclass, and nothing on the instance
      replaces one of the class's methods or the two functions it runs;
    - the sync slot holds a sync hook this package built, which the sync route always uses
      and the async route uses when the async slot is empty;
    - the async slot is empty, or holds the async hook this package built for the same
      policy and the same audit log.

    Anything else is answered False: a hook only in the async slot (the sync route then
    runs tools with no hook), a hook wrapped in another function, a hook from elsewhere, two
    hooks with different policies or logs, a subclass. Some of those may be fine; this
    cannot tell, so it does not say True. It reads the node as it is when called and says
    nothing about a later change to it, another node in the graph, or a tool called from
    outside a ToolNode.
    """
    if type(node) is not ToolNode:
        return False
    try:
        held = vars(node)
        sync_hook, async_hook = held["_wrap_tool_call"], held["_awrap_tool_call"]
        runs = (held["func"], ToolNode._func), (held["afunc"], ToolNode._afunc)
    except (TypeError, KeyError, AttributeError):  # not laid out as the ToolNode this knows
        return False
    for bound, function in runs:
        if getattr(bound, "__func__", None) is not function or bound.__self__ is not node:
            return False
    if any(callable(getattr(ToolNode, name, None)) for name in held):
        return False
    sync_info = _built(sync_hook)
    if sync_info is None or sync_info[0] != "sync":
        return False
    if async_hook is None:
        return True
    async_info = _built(async_hook)
    return (
        async_info is not None
        and async_info[0] == "async"
        and async_info[1] is sync_info[1]
        and async_info[2] is sync_info[2]
    )


def _built(hook: Any):
    """``(slot, policy, log)`` if ``hook`` is the very object this package built, else None.
    The lookup is by identity, so an object that only compares equal to a hook is not one."""
    entry = _HOOKS.get(id(hook))
    if entry is None or entry[0]() is not hook:
        return None
    return entry[1:]
