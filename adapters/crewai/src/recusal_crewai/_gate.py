"""The gate: CrewAI's own tool-call hooks, judging the invocation the tool receives.

The hook is CrewAI's own. CrewAI runs its ``before_tool_call`` hooks immediately before a
tool executes, with the tool name, the argument dict and the tool object; a hook that
returns ``False`` blocks the call. CrewAI then answers the model with a fixed line, "Tool
execution blocked by hook. Tool: <name>", and runs the ``after_tool_call`` hooks even on a
blocked call; an after hook that returns a string replaces that answer. So the gate is two
hooks that work as one: ``recusal_before_tool_call(policy)`` judges the call with
``recusal.decide`` and blocks a refusal, and ``recusal_after_tool_call(policy)``
replaces the fixed line with the reason the before hook blocked on, so the model reads why
on its next turn. ``install_recusal_hooks(policy)`` registers both and returns the function
that removes them.

The reason is the one the before hook decided, never a second judgment: a policy may be
stateful, and the blocked call must be explained by the decision that blocked it. The two
hooks correlate by the argument dict object itself: CrewAI hands the same ``tool_input``
dict to the before and after hooks of one invocation on one thread (verified for the sync
Crew runner, where every invocation also gets a fresh dict), so the before hook keeps the
reason beside a strong reference to that object in a small per-thread list, and the after
hook returns it only for that very object. Any other case, a different object, a different
thread, an entry already used or dropped, keeps CrewAI's generic line; a reason is never
borrowed from another call, and blocking never depends on the explanation being kept.

The policy judges the invocation the tool receives, not the dict the model proposed. CrewAI
validates the dict against the tool's ``args_schema`` at execution (``model_validate`` then
``model_dump``, on every path), which fills omitted defaults and coerces values, so a hook
that judged the proposal would judge ``{}`` for ``charge(amount: int = 100)`` while 100 ran.
The before hook therefore prepares the invocation first: omitted fields are completed from
the schema's defaults, the schema validates once, the validated values are written back
into the dict in place (the one CrewAI executes) and a copy is judged. So that this single
preparation is the one CrewAI repeats, the gate supports only a stock tool and schema: a
``CrewStructuredTool`` whose class and whose original ``BaseTool`` keep CrewAI's own run and
validation methods, and a schema that is a direct ``BaseModel`` subclass with no members of
its own beyond what Pydantic generates, configuration limited to ``extra`` and ``title``,
and every field a bare declaration (a supported annotation, an optional JSON default or a
``default_factory`` of ``list`` or ``dict``, an optional description and title; nothing
else) compared against the canonical ``FieldInfo``
that declaration produces. CrewAI's own generated schemas (``@tool`` functions, ``BaseTool``
subclasses without an ``args_schema``) are exactly that. A customised schema (a validator,
an overridden ``model_validate``, an intermediate base, a field with ``exclude``,
``exclude_if``, any other ``default_factory``, an alias or a constraint), an unsupported annotation
(``Any``, ``Annotated``, an ``Enum``, a model), an unknown parameter, a missing required one
and a validation failure are refused before judgment; a validation failure is reported by
field name only, never with Pydantic's text, which carries the input values.

The validated dict reaches the tool's callable as keyword arguments (``func`` on the
structured tool: the original ``BaseTool``'s ``_run``, or the application function a stock
``Tool`` wraps), and Python fills any parameter that callable declares beyond the schema from
the callable's own default, after judgment. The gate therefore holds the callable to the
schema: every named parameter it declares, and every one declared by a function a
``functools.wraps`` wrapper stands in front of, must be a schema field, or the call is
refused before judgment, naming the parameter and never its default. Declare it in the
schema.

CrewAI's hook registry is process-global: one install covers every agent, task and crew in
the process, including the coworker an agent delegates to, and every tool kind CrewAI runs
through its hooks (its own tools, ``@tool`` functions, MCP tools). Two consequences the tests
pin:

- hooks run in registration order and a later hook sees, and can change, the argument dict
  the gate judged. Install the gate AFTER your other hooks, so it judges the call exactly as
  it will run; a hook registered after it that rewrites the arguments bypasses the gate;
- the gate judges the tool name and arguments, not what a tool does once it runs. A tool
  that executes code or shell it is handed (CrewAI's code interpreter, for instance) needs a
  policy that names it; the gate cannot see inside.

The name the policy is given is the one CrewAI calls the tool by, which is the declared
name normalised (``sanitize_tool_name``: "Run SQL" becomes ``run_sql``), not the declared
name itself.

The tool the hook is handed is not always the tool CrewAI runs. CrewAI finds a tool by its
normalised name in more than one place, and the places disagree once two of the tools a
call is chosen from normalise alike: on the text (ReAct) route the hook's lookup keeps the
last of them while the executor runs the first; on the native routes the second is renamed
(``run_sql_2``), a name a third tool may hold, and the function that runs is looked up
apart from the tool the hook is handed. A hook's context carries neither the list the call
was chosen from nor the function about to run, and no list reachable from the agent or the
task is reliably that list (a task's tools replace the agent's, a crew adds and replaces
tools, an executor may be built once and kept). So the gate reads them where they are: in
the CrewAI function that entered the hooks and is about to run the tool, by its frame
(``_calling_site``). There it holds the handed tool to what will run
(``_identity_problem``), and refuses a call it cannot hold, including one CrewAI dispatches
from a function it does not know.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import inspect
import re
import sys
import threading
from types import FrameType, MethodType, MethodWrapperType, WrapperDescriptorType
from typing import Any, Callable, List, Optional

from crewai.agents.agent_executor import AgentExecutor
from crewai.hooks import (
    register_after_tool_call_hook,
    register_before_tool_call_hook,
    unregister_after_tool_call_hook,
    unregister_before_tool_call_hook,
)
from crewai.hooks.tool_hooks import ToolCallHookContext, run_before_tool_call_hooks
from crewai.tools import BaseTool
from crewai.tools.base_tool import Tool
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.utilities import agent_utils as _agent_utils
from crewai.utilities import tool_utils as _tool_utils
from crewai.utilities.string_utils import sanitize_tool_name
from pydantic import BaseModel, Field, ValidationError
from pydantic.fields import FieldInfo

from recusal import Finding, decide
from recusal.invocation import json_data as _json_data
from recusal.invocation import supported_annotation as _supported_annotation

#: A Recusal policy: ``policy(tool_name, tool_input)`` returns a list of findings.
Policy = Callable[[str, dict], List[Any]]

#: The answer CrewAI gives the model for a call a hook blocked; the after hook replaces it.
BLOCKED_PREFIX = "Tool execution blocked by hook."


#: The names Pydantic's metaclass (and the interpreter) put on a model class that declares
#: nothing of its own. Any other own name is a customisation: a method, a validator, a
#: property, a nested class, a plain attribute. ``tests/test_crewai_gate.py`` pins that a
#: stock ``create_model`` schema and a bare class statement stay within this set.
_GENERATED_NAMES = frozenset(
    {
        "__annotations__",
        "__module__",
        "__qualname__",
        "__doc__",
        "__dict__",
        "__weakref__",
        "__abstractmethods__",
        "_abc_impl",
        "model_config",
        "model_fields",
        "model_computed_fields",
        "__class_vars__",
        "__private_attributes__",
        "__signature__",
        "__firstlineno__",  # class statements on Python 3.13+
        "__static_attributes__",  # class statements on Python 3.13+
    }
)
_PYDANTIC_FAMILY = re.compile(r"^__pydantic_\w+__$")
_ALLOWED_CONFIG = frozenset({"extra", "title"})


def _canonical_field(field: FieldInfo) -> FieldInfo:
    """The ``FieldInfo`` the supported declaration produces: the annotation, a default when
    the field has one, and a description or title when the field sets one; nothing else."""
    extra: dict[str, Any] = {}
    if field.description is not None:
        extra["description"] = field.description
    if field.title is not None:
        extra["title"] = field.title
    # Pydantic types a field's annotation as optional; a field read from a model class has one
    annotation: Any = field.annotation
    if field.default_factory in _EMPTY_FACTORIES:
        return FieldInfo.from_annotated_attribute(
            annotation, Field(default_factory=field.default_factory, **extra)
        )
    if field.is_required():
        if extra:
            return FieldInfo.from_annotated_attribute(annotation, Field(**extra))
        return FieldInfo.from_annotation(annotation)
    return FieldInfo.from_annotated_attribute(annotation, Field(default=field.default, **extra))


#: The only ``default_factory`` values supported: each makes a new empty container, the same
#: on every call, so the default the gate completes and judges is the one that runs (the
#: gate writes it into the arguments CrewAI executes). CrewAI's own spreadsheet tool uses
#: ``list``.
_EMPTY_FACTORIES = (list, dict)


def _field_default(field: FieldInfo) -> Any:
    """The default an omitted field takes: its ``default``, or its empty factory's value."""
    if field.default_factory in _EMPTY_FACTORIES:
        return field.default_factory()  # type: ignore[call-arg]
    return copy.deepcopy(field.default)


def _unsupported_field(name: str, field: FieldInfo) -> Optional[str]:
    """Why this field is not a bare declaration, or None if it is."""
    if not _supported_annotation(field.annotation):
        return f"field `{name}` is not annotated with a supported data type"
    canonical = _canonical_field(field)
    # _attributes_set is every setting the declaration supplied; repr covers the metadata
    # list (constraints) and the value of each setting. Equal to the canonical form means
    # the declaration supplied nothing the contract does not name.
    if field._attributes_set != canonical._attributes_set or repr(field) != repr(canonical):
        return f"field `{name}` carries a setting beyond a default, description and title"
    if not field.is_required() and not _json_data(_field_default(field)):
        return f"field `{name}` has a default outside the JSON domain"
    return None


def _unsupported_schema(schema: Any) -> Optional[str]:
    """Why this argument schema is not a stock model, or None if it is."""
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        return "its argument schema is not a Pydantic model"
    if schema.__bases__ != (BaseModel,):
        return "its argument schema has a base class of its own"
    own = sorted(
        n for n in vars(schema) if n not in _GENERATED_NAMES and not _PYDANTIC_FAMILY.match(n)
    )
    if own:
        return f"its argument schema defines `{own[0]}`"
    decorators = getattr(schema, "__pydantic_decorators__", None)
    if dataclasses.is_dataclass(decorators) and any(
        getattr(decorators, f.name) for f in dataclasses.fields(decorators)
    ):
        return "its argument schema registers validators"
    config = sorted(set(schema.model_config) - _ALLOWED_CONFIG)
    if config:
        return f"its argument schema sets configuration `{config[0]}`"
    for name, field in schema.model_fields.items():
        why = _unsupported_field(name, field)
        if why is not None:
            return why
    return None


#: The structured tool's methods that validate and run an invocation on CrewAI's paths, and
#: the original tool's; the gate supports a tool only while every one of them is CrewAI's own.
_STOCK_STRUCTURED = ("_parse_args", "invoke", "ainvoke")
_STOCK_ORIGINAL = ("run", "arun", "_validate_kwargs", "to_structured_tool")


def _unsupported_tool(tool: Any) -> Optional[str]:
    """Why this tool's execution cannot be judged faithfully, or None if it can."""
    if not isinstance(tool, CrewStructuredTool):
        return "the call carries no CrewAI tool object"
    for name in _STOCK_STRUCTURED:
        if getattr(type(tool), name) is not getattr(CrewStructuredTool, name):
            return f"its tool class overrides `{name}`"
        if name in getattr(tool, "__dict__", {}):
            return f"its tool object replaces `{name}`"
    original = getattr(tool, "_original_tool", None)
    if original is not None:
        for name in _STOCK_ORIGINAL:
            # CrewAI's own method of that very name: another stock method in its place
            # (``run = BaseTool.arun``) is not stock for this one
            stock = {getattr(BaseTool, name), getattr(Tool, name)}
            if getattr(type(original), name, None) not in stock:
                return f"its tool class overrides `{name}`"
        # a stock ``Tool`` runs its ``func`` directly on the native routes and through
        # ``_run`` on the text route; a subclass with its own ``_run`` runs two things
        if isinstance(original, Tool):
            for name in ("_run", "_arun"):
                if getattr(type(original), name, None) is not getattr(Tool, name, None):
                    return f"its tool class overrides `{name}`"
        # a method set on the object itself hides the class's from every check above
        for name in (*_STOCK_ORIGINAL, "_run", "_arun"):
            if name in getattr(original, "__dict__", {}):
                return f"its tool object replaces `{name}`"
        # the native routes run the original tool, which validates with its own schema; the
        # gate prepares with the structured tool's, so they must be one
        if getattr(original, "args_schema", None) is not tool.args_schema:
            return "its argument schema is not its original tool's"
    why = _unsupported_schema(tool.args_schema)
    if why is not None:
        return why
    return _incomplete_callable(tool)


def _execution_callable(tool: Any) -> Any:
    """The application callable CrewAI finally runs for this structured tool.

    ``CrewStructuredTool.invoke`` calls ``func`` with the validated dict as keyword arguments.
    ``BaseTool.to_structured_tool`` sets ``func`` to the original tool's bound ``_run`` and
    keeps the original as ``_original_tool``; a stock ``Tool`` (what ``@tool`` builds)
    forwards ``_run(*args, **kwargs)`` to the application function it wraps, whose signature
    is the one Python completes. Raises ``TypeError`` when ``func`` is not the original's
    ``_run``: the gate cannot say what would run.
    """
    func = tool.func
    original = getattr(tool, "_original_tool", None)
    if original is None:
        return func
    if func != original._run:
        raise TypeError("its callable is not the original tool's `_run`")
    if type(original)._run is Tool._run:
        return original.func
    return func


class _DeclaredSignature(Exception):
    """A callable on the way to the code that runs carries a ``__signature__`` of its own."""


#: Wrappers that pass every argument through unchanged and declare no parameters of their
#: own, so the code that runs is the function they wrap: ``functools.lru_cache`` and
#: ``functools.cache``.
_PASS_THROUGH = (type(functools.lru_cache(maxsize=None)(lambda: None)),)

#: A ``__call__`` implemented in C (a slot wrapper or a bound one): it carries no
#: ``__signature__`` of its own and has nothing further in to read. A bound method is one
#: such callable: its own ``__dict__`` shows its function's attributes, so a signature
#: declared there is found above, and its ``__call__`` is where the walk stops. Walking
#: into C slot wrappers instead ran out of depth and refused every ordinary tool method.
_C_CALL = (WrapperDescriptorType, MethodWrapperType)


def _own_parameters(func: Any, depth: int = 0) -> Any:
    """The parameters Python binds when ``func`` is called: the callable's own, never a
    signature it shows for something else.

    ``inspect.signature`` follows ``__wrapped__`` (a ``functools.wraps`` wrapper would show
    the wrapped function's parameters while its own defaults run) and honours a declared
    ``__signature__``, on the callable or one level in (a callable object's ``__call__``, a
    ``functools.partial``'s function; a method shows its function's attributes as its
    own). The first is turned off; the second raises ``_DeclaredSignature`` wherever it is
    found. Wrappers that only pass arguments through (``lru_cache``, ``cache``, a
    ``staticmethod``, and ``lru_cache`` over a method) are looked through.
    """
    if depth > 8:
        raise ValueError("too many wrappers to inspect")
    if isinstance(func, staticmethod):
        return _own_parameters(func.__func__, depth + 1)
    if isinstance(func, _PASS_THROUGH):
        return _own_parameters(func.__wrapped__, depth + 1)
    if inspect.ismethod(func) and isinstance(func.__func__, _PASS_THROUGH):
        return _own_parameters(_bound(func.__func__.__wrapped__, func.__self__), depth + 1)
    _refuse_declared_signature(func, depth)
    return inspect.signature(func, follow_wrapped=False).parameters


def _bound(function: Any, owner: Any) -> Any:
    """``function`` bound to ``owner`` as a method lookup would bind it, so its first
    parameter is not read as one the caller passes; anything else is returned as it is."""
    return MethodType(function, owner) if inspect.isfunction(function) else function


def _wrapped_parameters(func: Any) -> list:
    """The parameters of each callable a ``functools.wraps`` chain stands in front of.

    A wrapper that takes ``(*args, **kwargs)`` passes the call to the function it wraps,
    which fills its own defaults: a parameter declared there and not in the schema runs
    with a value the policy never sees, though the wrapper's own parameters show none.
    """
    owner = func.__self__ if inspect.ismethod(func) else None
    current = func.__func__ if inspect.ismethod(func) else func
    found: list[Any] = []
    for _ in range(8):
        current = getattr(current, "__wrapped__", None)
        if current is None:
            return found
        found.append(_own_parameters(current if owner is None else _bound(current, owner)))
    raise ValueError("too many wrappers to inspect")


def _refuse_declared_signature(func: Any, depth: int) -> None:
    """Raise ``_DeclaredSignature`` if ``func``, or what ``inspect.signature`` would read
    in its place, declares ``__signature__``."""
    if depth > 8:
        raise ValueError("too many wrappers to inspect")
    if "__signature__" in getattr(func, "__dict__", {}) or hasattr(type(func), "__signature__"):
        raise _DeclaredSignature
    if isinstance(func, functools.partial):
        _refuse_declared_signature(func.func, depth + 1)
    elif not (inspect.isfunction(func) or inspect.isbuiltin(func) or isinstance(func, type)):
        call = getattr(type(func), "__call__", None)
        if call is not None and not isinstance(call, _C_CALL):
            _refuse_declared_signature(call, depth + 1)


def _incomplete_callable(tool: Any) -> Optional[str]:
    """Why the callable declares a parameter the schema does not, or None if it declares
    none. Python would fill such a parameter from the callable's default after judgment, so
    the policy would never see the value that runs; the refusal names the parameter and
    never its default."""
    try:
        func = _execution_callable(tool)
    except TypeError as exc:
        return str(exc)
    try:
        levels = [_own_parameters(func), *_wrapped_parameters(func)]
    except _DeclaredSignature:
        return "its callable declares a signature that is not its own"
    except (TypeError, ValueError):
        return "its callable cannot be inspected for its parameters"
    fields = tool.args_schema.model_fields
    for parameters in levels:
        for name, parameter in parameters.items():
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                continue
            if parameter.kind is parameter.POSITIONAL_ONLY:
                return f"its callable's parameter `{name}` is positional-only"
            if name not in fields:
                return f"its callable's parameter `{name}` is absent from the argument schema"
    return None


def _prepared(
    tool: Any, tool_input: dict[str, Any]
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Prepare the invocation the tool will execute, in place, and return a detached copy
    for judgment; or ``(None, reason)`` when it cannot be prepared faithfully.

    Omitted fields are completed from the schema's defaults into the raw input, the schema
    validates once (the same ``model_validate`` CrewAI runs at execution, which coerces an
    explicit ``"100"`` to 100 for an ``int`` field), and the validated values replace the
    dict's contents, so the dict CrewAI executes is the prepared invocation and the policy
    judges a copy of it. A default that is not a valid value of its own field is left for
    the schema to supply, as CrewAI's own validation supplies it. A field the schema
    requires and the input omits is refused, as is an unknown parameter and a validation
    failure (named by field, without Pydantic's text).
    """
    why = _unsupported_tool(tool)
    if why is not None:
        return None, f"its arguments cannot be judged ({why})"
    fields = tool.args_schema.model_fields
    unknown = sorted(k for k in tool_input if k not in fields)
    if unknown:
        return None, f"`{unknown[0]}` is not a parameter of the tool"
    raw: dict[str, Any] = {}
    for name, field in fields.items():
        if name in tool_input:
            raw[name] = tool_input[name]
        elif not field.is_required():
            raw[name] = _field_default(field)
        else:
            return None, f"required parameter `{name}` is missing"
    prepared, failed = _validated(tool.args_schema, raw)
    left_out = sorted(failed & {name for name in fields if name not in tool_input})
    if prepared is None and failed and len(left_out) == len(failed):
        # Only defaults failed: a default that is not a valid value of its own field
        # (``limit: int = None``, as CrewAI generates for ``def f(limit: int = None)``).
        # Pydantic does not validate a default, so CrewAI runs it as it stands. Leave it
        # out of the dict CrewAI executes, for the schema to supply exactly as CrewAI's own
        # validation does, and judge what that validation produces.
        for name in left_out:
            del raw[name]
        if not raw:
            # the text route invokes an empty call with no arguments at all, so the
            # schema's default would not be the one that runs
            return None, (
                f"the default of `{left_out[0]}` is not a valid value of its own field, and "
                "a call with no other arguments cannot carry it; declare the field Optional"
            )
        prepared, failed = _validated(tool.args_schema, raw)
    if prepared is None:
        named = sorted(failed)
        return None, f"validation failed for `{named[0]}`" if named else "validation failed"
    tool_input.clear()
    tool_input.update({name: prepared[name] for name in raw})
    # the policy gets its own copy, nested values included: ``decide`` hands the policy the
    # object it is given, and a policy that edits it must not edit the call that runs
    return copy.deepcopy(prepared), None


def _validated(schema: Any, raw: dict[str, Any]) -> tuple[Optional[dict[str, Any]], set]:
    """``schema.model_validate(raw).model_dump()``, the validation CrewAI runs before the
    tool, or ``(None, the fields that failed)``."""
    try:
        return schema.model_validate(raw).model_dump(), set()
    except ValidationError as exc:
        return None, {str(e["loc"][0]) for e in exc.errors() if e.get("loc")}
    except Exception:  # noqa: BLE001, a schema that fails otherwise cannot be judged
        return None, set()


def _code(function: Any) -> Any:
    """A function's own code object, looking through ``functools.wraps`` wrappers (tracing
    and instrumentation wrap CrewAI's functions; the frame that runs is the original's)."""
    return inspect.unwrap(function).__code__


#: CrewAI's entry to its before-tool-call hooks. Whatever calls it is about to run a tool.
_DISPATCH_ENTRY = _code(run_before_tool_call_hooks)

#: The CrewAI functions that call that entry and then run the tool, by code object. The
#: text route holds the tools the call is chosen from in its local ``tools``, and the name
#: it selects by and the arguments it executes in ``tool_calling``. The native routes hold
#: the name in ``func_name``, the arguments in ``args_dict`` and the functions that run in
#: ``self._available_functions`` (the agent executor) or ``available_functions`` (the
#: plan-and-execute step). tests/test_crewai_gate.py pins each of these names.
_TEXT_SITE = _code(_tool_utils.execute_tool_and_check_finality)
_NATIVE_EXECUTOR_SITE = _code(AgentExecutor._execute_single_native_tool_call)
_NATIVE_STEP_SITE = _code(_agent_utils.execute_single_native_tool_call)

_UNKNOWN_SITE = "unknown"


def _entry_wrappers() -> frozenset:
    """The code of every wrapper that now stands, in one of the modules CrewAI calls its
    hook entry from, between that module and the entry. Tracing and instrumentation
    replace the module's name with such a wrapper; its frame then sits between the entry
    and the function running the call, and is walked past. Two kinds are read: a
    ``functools.wraps`` wrapper, whose own code runs, and a ``wrapt`` function wrapper (as
    OpenTelemetry instrumentors install), which is C and leaves no frame of its own but
    calls the instrumentor's wrapper function, whose code runs. Nothing else is walked
    past: a frame of any other code between the two is a caller the gate does not know."""
    codes = set()
    for module in (
        _tool_utils,
        _agent_utils,
        sys.modules.get(AgentExecutor.__module__),
        sys.modules.get(run_before_tool_call_hooks.__module__),
    ):
        current = getattr(module, "run_before_tool_call_hooks", None)
        for _ in range(8):
            if current is None:
                break
            # read wrapt's own attribute first: a wrapt proxy forwards ``__code__`` to the
            # function it wraps, which is not the code that stands between
            instrumentor = getattr(current, "_self_wrapper", None)
            if instrumentor is not None:
                code = getattr(getattr(instrumentor, "__func__", instrumentor), "__code__", None)
            elif hasattr(current, "__wrapped__"):
                code = getattr(current, "__code__", None)
            else:
                break
            if code is not None:
                codes.add(code)
            current = getattr(current, "__wrapped__", None)
    return frozenset(codes)


def _calling_site() -> Optional[tuple[str, Any]]:
    """The CrewAI function that is running this tool call: ``(kind, its local variables)``.

    Walks up the stack to CrewAI's entry to its before hooks
    (``run_before_tool_call_hooks``); the function that called it is the one about to run
    the tool. Frames between the hook and that entry are walked past, whoever's they are
    (CrewAI's dispatch, a filter CrewAI wraps a hook in, a hook of yours that calls the
    gate). ``("unknown", None)`` when that caller is not a function the gate knows, or when
    CrewAI's hook dispatch is on the stack without that entry. ``None`` when neither is on
    the stack: the hook was called by hand, and CrewAI is not about to run anything for it.
    """
    dispatching = False
    frame: Optional[FrameType] = sys._getframe(1)
    while frame is not None:
        if frame.f_code is _DISPATCH_ENTRY:
            caller = frame.f_back
            wrappers = _entry_wrappers()
            while caller is not None and caller.f_code in wrappers:
                caller = caller.f_back
            if caller is None:
                return _UNKNOWN_SITE, None
            if caller.f_code is _TEXT_SITE:
                return "text", caller.f_locals
            if caller.f_code is _NATIVE_EXECUTOR_SITE:
                return "native-executor", caller.f_locals
            if caller.f_code is _NATIVE_STEP_SITE:
                return "native-step", caller.f_locals
            return _UNKNOWN_SITE, None
        if str(frame.f_globals.get("__name__", "")).startswith("crewai.hooks"):
            dispatching = True
        frame = frame.f_back
    return (_UNKNOWN_SITE, None) if dispatching else None


_MISSING: Any = object()


def _unbound_arguments(site: Optional[tuple[str, Any]], tool_input: Any) -> Optional[str]:
    """Make the dict the hook was given the dict CrewAI executes, or say why it is not.

    On the text route CrewAI executes ``tool_calling.arguments``. That is the hook's dict,
    unless a hook replaced ``tool_input`` with another (refused) or the model sent no
    arguments: then the hook gets a new empty dict and the
    tool is invoked with nothing, so the invocation this gate prepares in the hook's dict
    (and anything an earlier hook wrote there) would be judged and thrown away. The gate
    puts the hook's dict in its place before judging; preparation then fills that one.
    The native routes call the tool with their own ``args_dict``, which is the hook's dict
    unless a hook replaced ``tool_input`` with another; then the call is refused, since the
    gate would judge one dict while another runs.
    """
    if site is None:
        return None
    if site[0] in ("native-executor", "native-step"):
        if site[1].get("args_dict", _MISSING) is not tool_input:
            return "the arguments handed to the gate are not the ones CrewAI would run"
        return None
    if site[0] != "text":
        return None
    calling = site[1].get("tool_calling", _MISSING)
    if calling is _MISSING:
        return "the gate cannot read the arguments CrewAI is about to run this call with"
    if getattr(calling, "arguments", _MISSING) is tool_input:
        return None
    if not isinstance(tool_input, dict) or getattr(calling, "arguments", _MISSING):
        return "the arguments handed to the gate are not the ones CrewAI would run"
    try:
        calling.arguments = tool_input
    except Exception:  # noqa: BLE001, an object that cannot be set leaves the call unjudgeable
        pass
    if getattr(calling, "arguments", _MISSING) is not tool_input:
        return "the arguments handed to the gate are not the ones CrewAI would run"
    return None


def _same_execution(candidate: Any, tool: Any) -> bool:
    """Whether running ``candidate`` is running ``tool``: two structured tools built from
    one original tool (a tool listed twice), of one class, with the one schema and the one
    callable. Which of the two CrewAI picks then makes no difference."""
    original = getattr(tool, "_original_tool", None)
    return (
        original is not None
        and getattr(candidate, "_original_tool", None) is original
        and type(candidate) is type(tool)
        and getattr(candidate, "args_schema", None) is tool.args_schema
        and getattr(candidate, "func", None) == tool.func
    )


def _identity_problem(context: Any, site: Optional[tuple[str, Any]]) -> Optional[str]:
    """Why the tool this hook was handed may not be the tool CrewAI runs, or None.

    Refused, before judgment:

    - a call whose name is not the handed tool's own normalised name: CrewAI renamed the
      tool because another holds its name, and a rule written for the declared name would
      not match the new one;
    - on the text route, a call unless the name CrewAI selects the tool by
      (``tool_calling.tool_name``) is the call's, the handed tool is among the tools it is
      chosen from, and every tool of that name there is the handed tool or a copy of it
      (``_same_execution``). CrewAI's executor then selects that tool or its copy;
    - on the native routes, a call unless the function CrewAI runs for that name is the
      handed tool's original tool's ``run``, the class's own;
    - a call CrewAI dispatches from a function the gate does not know.

    A hook called by hand, outside CrewAI's hook dispatch and with no agent on its
    context, is held to the first only. Every CrewAI function that runs a tool passes its
    agent, so a context with an agent and no CrewAI function to be found (a hook that runs
    the gate on another thread) is refused.
    """
    called = context.tool_name
    if not isinstance(called, str):
        return "the call carries no tool name"
    tool = getattr(context, "tool", None)
    if tool is None:
        return None  # refused by ``_unsupported_tool``: the call carries no tool object
    declared = getattr(tool, "name", None)
    if not isinstance(declared, str) or sanitize_tool_name(declared) != sanitize_tool_name(called):
        return (
            f"CrewAI handed the gate a tool of another name for a call to `{called}`, so the "
            "gate cannot say which tool would run; CrewAI renames a tool whose name another "
            "tool shares, so give each tool a distinct name"
        )
    if site is None:
        if getattr(context, "agent", None) is not None:
            return (
                "the gate cannot find the CrewAI function running this call (it must run on "
                "the thread CrewAI runs its hooks on), so it cannot say which tool would run"
            )
        return None
    kind, local = site
    if kind == "text":
        tools = local.get("tools", _MISSING)
        if tools is _MISSING or tools is None:
            return "the gate cannot read the tools CrewAI is choosing this call from"
        # CrewAI runs the tool it selects by ``tool_calling.tool_name``, not the one on the
        # context, which an earlier hook may have changed
        running = getattr(local.get("tool_calling"), "tool_name", None)
        if not isinstance(running, str) or sanitize_tool_name(running) != sanitize_tool_name(
            called
        ):
            return f"the call handed to the gate for `{called}` is not the one CrewAI would run"
        named = [
            candidate
            for candidate in tools
            if sanitize_tool_name(str(getattr(candidate, "name", ""))) == sanitize_tool_name(called)
        ]
        if not any(candidate is tool for candidate in named):
            return f"the tool handed to the gate for `{called}` is not the one CrewAI would run"
        if not all(candidate is tool or _same_execution(candidate, tool) for candidate in named):
            return (
                f"{len(named)} of the tools CrewAI is choosing this call from are named "
                f"`{sanitize_tool_name(called)}` once it normalises their names, so the gate "
                "cannot say which would run; give each tool a distinct name"
            )
        return None
    if kind in ("native-executor", "native-step"):
        if kind == "native-executor":
            functions = getattr(local.get("self"), "_available_functions", _MISSING)
        else:
            functions = local.get("available_functions", _MISSING)
        name = local.get("func_name", _MISSING)
        if not isinstance(functions, dict) or name != called:
            return "the gate cannot read the function CrewAI is about to run for this call"
        original = getattr(tool, "_original_tool", None)
        runs = functions.get(called)
        if (
            original is None
            or runs is None
            or getattr(runs, "__self__", None) is not original
            or getattr(runs, "__func__", None) is not getattr(type(original), "run", None)
        ):
            return (
                f"the tool handed to the gate for `{called}` is not the one CrewAI would run; "
                "if two tools share a name once CrewAI normalises it, give each a distinct name"
            )
        return None
    return (
        "CrewAI ran its hooks from a function this version of recusal-crewai does not "
        "know, so it cannot say which tool would run"
    )


#: Given the hook's ``ToolCallHookContext``, the recording arguments ``decide`` takes for the
#: call (``audit=``, ``surface=``, ...). examples/crewai_audit.py builds one.
Recording = Callable[[Any], dict]


def _unjudgeable(tool_name: str, detail: str, proposal: Any, record: dict) -> str:
    """The reason for a call the gate refuses before any policy runs.

    Without recording arguments it is the plain text. With them (examples/crewai_audit.py)
    the refusal goes through ``decide`` like every other, so it is one entry on the log,
    bound to the arguments as the model proposed them, and the reason kept for the model is
    the one on the record.
    """
    if not record:
        return f"Recusal refused `{tool_name}`: {detail}"
    bound = proposal if isinstance(proposal, dict) else {"input": proposal}

    def unjudgeable(_name, _input):
        return [Finding.fail("recusal_unjudgeable_input", severity="CRITICAL", message=detail)]

    try:
        _, reason = decide(tool_name, bound, unjudgeable, safe_errors=True, **record)
    except Exception as exc:  # noqa: BLE001, the refusal stands even if the record cannot be written
        return f"Recusal failed closed (policy error): RECUSAL_POLICY_ERROR ({type(exc).__name__})"
    return reason


def _refusal(
    tool: Any, tool_name: str, tool_input: Any, policy: Policy, **record: Any
) -> Optional[str]:
    """Return the reason a call is refused, or None to run it.

    The policy judges a copy of the prepared invocation (``_prepared``), which is also what
    the dict CrewAI executes now holds. A non-dict is refused: the policy cannot read it.

    ``record`` goes straight to ``decide``, so a gate that puts the decision on a log
    (examples/crewai_audit.py) decides, and refuses, exactly as this one does. The entry
    for a judged call carries the fingerprint of the prepared invocation CrewAI executes;
    a call refused before judgment is bound to the arguments as proposed (``_prepared``
    changes the dict only when it succeeds).
    """
    if not isinstance(tool_input, dict):
        return _unjudgeable(tool_name, "arguments are not a mapping", tool_input, record)
    judged, why = _prepared(tool, tool_input)
    if judged is None:
        return _unjudgeable(tool_name, str(why), tool_input, record)
    try:
        decision, reason = decide(tool_name, judged, policy, safe_errors=True, **record)
    except Exception as exc:  # noqa: BLE001, a buggy policy must not silently disable the gate
        return f"Recusal failed closed (policy error): RECUSAL_POLICY_ERROR ({type(exc).__name__})"
    return reason if decision == "deny" else None


#: How many blocked invocations one thread may hold reasons for before the oldest is
#: dropped (a dropped entry degrades to CrewAI's generic line, never to a wrong reason).
PENDING_LIMIT = 16


class _Denials:
    """The reasons the before hook blocked on, per thread, keyed by the argument object.

    One entry per blocked invocation: ``(the tool_input object, reason)``. The object is
    held strongly so its id cannot be reused while the entry lives. The before hook first
    drops any entry for the same object, whatever the decision: one object cannot carry two
    live invocations on one thread, so an entry still present is from an invocation whose
    after hook never ran, and must not be attached to the next call that reuses the dict.
    """

    def __init__(self):
        self._local = threading.local()

    def _entries(self) -> list:
        entries = getattr(self._local, "entries", None)
        if entries is None:
            entries = self._local.entries = []
        return entries

    def forget(self, tool_input: Any) -> None:
        entries = self._entries()
        entries[:] = [e for e in entries if e[0] is not tool_input]

    def keep(self, tool_input: Any, reason: str) -> None:
        entries = self._entries()
        entries.append((tool_input, reason))
        del entries[:-PENDING_LIMIT]

    def take(self, tool_input: Any) -> Optional[str]:
        entries = self._entries()
        for index, (held, reason) in enumerate(entries):
            if held is tool_input:
                del entries[index]
                return reason
        return None


def recusal_before_tool_call(
    policy: Policy,
    denials: Optional[_Denials] = None,
    recording: Optional[Recording] = None,
):
    """Return a ``before_tool_call`` hook: ``False`` (block) on a refusal, ``None`` to run.

    ``denials`` is shared with the matching after hook (``install_recusal_hooks`` builds the
    pair); without it the reason is not kept and the after hook keeps CrewAI's line.
    ``recording`` puts every decision on an audit log; a ``recording`` that raises blocks
    the call, like any other failure inside the hook.
    """
    store = denials

    def gate(context: ToolCallHookContext) -> Optional[bool]:
        try:
            tool_input = context.tool_input
            if store is not None:
                try:
                    store.forget(tool_input)
                except Exception:  # noqa: BLE001, bookkeeping never changes the decision
                    pass
            record = dict(recording(context)) if recording is not None else {}
            site = _calling_site()
            problem = _identity_problem(context, site) or _unbound_arguments(site, tool_input)
            if problem is not None:
                reason: Optional[str] = _unjudgeable(
                    str(context.tool_name), problem, tool_input, record
                )
            else:
                reason = _refusal(
                    getattr(context, "tool", None), context.tool_name, tool_input, policy, **record
                )
            if reason is None:
                return None
            if store is not None:
                try:
                    store.keep(tool_input, reason)
                except Exception:  # noqa: BLE001, a failure to keep the reason cannot unblock
                    pass
            return False
        except Exception:  # noqa: BLE001, CrewAI swallows a hook's exception and runs the tool
            return False

    gate.recusal_gate = True  # type: ignore[attr-defined]
    return gate


def recusal_after_tool_call(policy: Policy, denials: Optional[_Denials] = None):
    """Return an ``after_tool_call`` hook that puts the before hook's reason in a blocked
    call's answer.

    The reason is the one kept for this invocation's argument object; the call is never
    judged again. A call another hook blocked, or one whose reason was not kept (another
    thread, a dropped entry, no shared ``denials``), keeps CrewAI's own line.
    """
    store = denials

    def explain(context: ToolCallHookContext) -> Optional[str]:
        if not str(context.tool_result).startswith(BLOCKED_PREFIX):
            return None
        if store is None:
            return None
        try:
            return store.take(context.tool_input)
        except Exception:  # noqa: BLE001, the generic line is the safe fallback
            return None

    explain.recusal_gate = True  # type: ignore[attr-defined]
    return explain


def install_recusal_hooks(
    policy: Policy, recording: Optional[Recording] = None
) -> Callable[[], None]:
    """Register the gate's two hooks for every tool call in this process; return ``uninstall``.

    Register it after your other hooks: hooks run in registration order, and a hook that
    runs after the gate can change the arguments the gate judged. ``recording`` puts every
    decision on an audit log (examples/crewai_audit.py).
    """
    denials = _Denials()
    before = recusal_before_tool_call(policy, denials, recording)
    after = recusal_after_tool_call(policy, denials)
    register_before_tool_call_hook(before)
    register_after_tool_call_hook(after)

    def uninstall() -> None:
        unregister_before_tool_call_hook(before)
        unregister_after_tool_call_hook(after)

    return uninstall
