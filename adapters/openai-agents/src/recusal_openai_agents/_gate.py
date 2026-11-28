"""The gate: a Recusal tool input guardrail on every function tool of an agent.

The hook is the SDK's own: a tool input guardrail runs immediately before a function tool
executes, and ``recusal_guardrail(policy)`` judges the call with ``recusal.decide``. A
refusal becomes the call's output, so the model sees Recusal's reason on its next turn and
the tool never runs. Every call in a model turn is judged separately, and ``Runner.run``,
``Runner.run_streamed`` and a human approval (``needs_approval``) all pass through it: the
guardrail runs when the tool is about to execute, after any approval.

What the guardrail cannot see, so ``gated_agent`` refuses it rather than skipping it:

- every tool that is not a ``FunctionTool``: hosted tools that run on OpenAI's servers (web
  search, file search, code interpreter, image generation, hosted MCP, tool search) and
  local ones the SDK executes itself (shell, local shell, apply patch, computer, custom,
  programmatic tool calling). The rule is default-deny: anything but a ``FunctionTool`` is
  refused, so a tool type added in a later SDK release is refused too;
- MCP servers attached with ``mcp_servers=``, and handoffs to other agents;
- an agent with a ``prompt``: a prompt stored on OpenAI's servers can configure tools of
  its own, which are not in the agent's ``tools`` list;
- an agent whose ``model_settings`` put tools into the request themselves (``extra_body``
  or ``extra_args`` with a ``tools``, ``prompt``, ``plugins``, ``web_search_options`` or
  ``mcp_servers`` key);
- a sub-agent behind ``agent.as_tool(...)`` whose own tools are not gated: the guardrail
  sees only the input handed to the sub-agent, never the calls it then makes. Gate the
  sub-agent first, ``gated_agent(sub, policy).as_tool(...)``, and the parent accepts it.
  What the parent's policy is handed for such a call is the input the sub-agent is
  given (:func:`_agent_tool_input`); an ``as_tool(parameters=...)`` type the gate cannot
  work that out for is refused (:func:`_agent_tool_refusal`);
- the experimental Codex tool, which runs its own commands;
- ``Agent`` subclasses such as ``SandboxAgent``, which add tools of their own at run time.

What the guardrail could never judge, so ``gate_tools`` and ``gated_agent`` refuse the tool
when you call them, naming the tool and the cause (:func:`_tool_refusal`):

- a parameter the SDK prepares only by running the tool's own code (an ``Enum``, a model, a
  dataclass, a ``TypedDict``, a date, a tuple, a set, a ``Sequence`` or an ``Iterable``,
  ``bytes``, a string pattern, a custom validator, a default factory other than ``list``
  or ``dict``, a default that is not JSON data): every call would be refused as
  unjudgeable;
- a mapping parameter whose key type is neither ``str`` nor a ``Literal`` of strings: the
  SDK converts the keys, and arguments are judged as JSON data, whose keys are strings;
- a variadic parameter (``*args``, ``**kwargs``), whose values have no names to judge by;
- a wrapper around the function that takes a parameter of its own, or takes one of the
  function's: it would run with a value no policy judged;
- a ``functools.singledispatch`` function or a ``singledispatchmethod``, with or without
  an implementation registered, wherever it stands among the callables read for that
  (below): it picks the code that runs at call time by the type of an argument, and an
  implementation can be registered after a call was judged and before it runs;
- a ``functools.partialmethod`` read from a class, or from an object when the callable
  it holds has no ``__get__`` (a class-based decorator's instance; a ``functools.partial``
  up to Python 3.13): what runs is the callable it holds, which ``inspect`` reads past;
- a ``weakref.proxy``, which shows the attributes of the object it stands for and is not
  that object;
- a method, or a ``functools.partial`` with positional arguments, over a wrapper of a
  function whose leading parameter is not named ``self`` or ``cls``: the SDK leaves that
  parameter out and passes the judged values by position, so a wrapper that drops what
  is bound runs every value one place off (:func:`_kept_receiver`);
- a decorator whose signature cannot be read, and a function under more wrappers than
  the walk reads;
- what the kernel's walk (``recusal.invocation.reached_parameters``, ``recusal`` 0.17.0
  or later) refuses by a class of its own, each with a reason that says what was found
  and what to do (:func:`_unread_reason`): a decorator that holds a second wrapper of
  the function, where that wrapper takes a parameter the walk cannot account for, takes
  the tool's parameters in another order, or the way from it cannot be judged
  (``HiddenWrapper``), a ``functools.lru_cache`` wrapper whose function cannot be told
  (``UnreadCache``), a bound argument the walk cannot follow through a wrapper
  (``BoundArgumentKept``), a declared signature (``DeclaredSignature``), a
  single-dispatch function with nothing registered (``OpenDispatch``), and a callable
  that answers for another object (``ProxiedCallable``; a weak proxy, ``WeakProxy``,
  gets the reason this gate gives the ones it finds). Some tools refused for these run
  correctly, and each finds exact shapes and not every one: the package page lists what
  is refused although correct and what is not found.

Which callables are read for that: the function, what its ``__wrapped__`` chain leads to,
the function a ``functools.lru_cache`` wrapper calls (``recusal.invocation.cache_target``),
which is not the one it names where ``functools.wraps`` was written above the cache, a
bound method's function, a callable object's ``__call__`` and the callable a
``functools.partialmethod`` holds. So the three questions this gate asks itself (a
single-dispatch function, a partialmethod, a weak proxy) are asked of what such a cache
calls too; 0.2.2 asked them of what it names alone. A function a wrapper holds some
other way (in a closure, a global, an attribute) is not walked: the kernel's walk refuses
the shapes of it named above and states the rest as not found (``recusal.invocation``).

``gated_agent`` checks the agent when you call it. Build the agent completely, then gate
it; a tool appended to the gated agent's list afterwards is not gated. The gated copy has
its own ``tools``, ``handoffs`` and ``mcp_servers`` lists, so a later change to the
original agent's lists does not reach it.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import json
import types
import weakref
from collections.abc import Mapping
from typing import Any, Callable, List, Optional

from agents import (
    Agent,
    FunctionTool,
    ToolGuardrailFunctionOutput,
    tool_input_guardrail,
)
from agents._function_tool_arguments import _is_data_only_schema, can_prepare_without_user_code
from agents.agent_tool_input import AgentAsToolInput
from agents.function_schema import FuncSchema

from recusal import Finding, decide
from recusal.invocation import BoundArgumentKept as _BoundArgumentKept
from recusal.invocation import DeclaredSignature as _DeclaredSignature
from recusal.invocation import DispatchedAtCallTime as _DispatchedAtCallTime
from recusal.invocation import HiddenWrapper as _HiddenWrapper
from recusal.invocation import OpenDispatch as _OpenDispatch
from recusal.invocation import PartialMethodSignature as _PartialMethodSignature
from recusal.invocation import ProxiedCallable as _ProxiedCallable
from recusal.invocation import UnreadCache as _UnreadCache
from recusal.invocation import WeakProxy as _WeakProxy
from recusal.invocation import cache_target as _cache_target
from recusal.invocation import detached as _detached
from recusal.invocation import reached_parameters as _reached_parameters

#: A Recusal policy: ``policy(tool_name, tool_input)`` returns a list of findings.
Policy = Callable[[str, dict], List[Any]]
#: The name the guardrail carries, which is how a gated tool is recognized.
GUARDRAIL_NAME = "recusal"


_NOT_BUILT_BY_FUNCTION_TOOL = (
    "its invoker does not expose the arguments it would execute, so the gate could judge "
    "only the model's raw JSON. Build the tool with function_tool."
)

_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)

#: How many levels below the arguments object a value may sit. A fixed number, the same
#: with and without a recording: nothing nested deeper is copied, fingerprinted, prepared
#: or handed to a policy, each of which walks the arguments and would give up at a depth
#: that depends on the interpreter's recursion limit and the caller's stack.
_MAX_DEPTH = 64

_TOO_DEEP = "arguments are nested too deeply"

#: For arguments that did not arrive as text. The braces take the name of their type
#: (:func:`_type_name`), never anything they hold.
_NOT_TEXT = (
    "arguments are not JSON text (they arrived as a `{}`, which is not exactly `str`, "
    "`bytes` or `bytearray`), so they cannot be judged"
)


def _type_name(value: Any) -> str:
    """The name of ``value``'s type, for a reason and a record: the name itself where it
    is a plain identifier, and ``object`` for any other name a class was given, so that
    nothing but a word reaches the text."""
    name = type.__dict__["__name__"].__get__(type(value))
    return name if type(name) is str and name.isidentifier() else "object"


def _too_deep(value: Any, depth: int = 0) -> bool:
    """Whether decoded arguments hold a value more than ``_MAX_DEPTH`` levels down."""
    if depth > _MAX_DEPTH:
        return True
    if type(value) is dict:
        return any(_too_deep(item, depth + 1) for item in value.values())
    if type(value) is list:
        return any(_too_deep(item, depth + 1) for item in value)
    return False


def _sdk_schema(tool: FunctionTool) -> FuncSchema:
    """The function schema the SDK built for this tool and executes by: the parameters it
    binds the validated arguments to, and whether the first is the context it fills itself
    (``takes_context``).

    Read from the SDK's own preparation function, which holds it, and never re-derived
    from the function's annotations: the SDK resolves a context annotation written as a
    string, through an alias or under another import name, and a second reading that
    disagreed would name every argument one parameter off. Raises ``TypeError`` for a tool
    that carries none (one not built by ``function_tool``)."""
    implementation = getattr(tool.on_invoke_tool, "_invoke_tool_impl", None)
    prepare: Any = getattr(implementation, "__agents_prepare_arguments__", None)
    try:
        schema = inspect.getclosurevars(prepare).nonlocals.get("schema")
    except TypeError:
        schema = None
    if not isinstance(schema, FuncSchema) or not callable(
        getattr(tool.on_invoke_tool, "prepare_arguments", None)
    ):
        raise TypeError(_NOT_BUILT_BY_FUNCTION_TOOL)
    return schema


def _judged_names(schema: FuncSchema) -> tuple[list[str], list[str]]:
    """The names of the parameters the SDK passes by position and by keyword, in the order
    and by the rule of the SDK's own ``FuncSchema.to_call_args``: the first parameter is
    left out when the SDK says it is the context. Raises ``TypeError`` for a variadic
    parameter, whose values have no name."""
    positional: list[str] = []
    keyword: list[str] = []
    for index, parameter in enumerate(schema.signature.parameters.values()):
        if schema.takes_context and index == 0:
            continue
        if parameter.kind in _VARIADIC:
            stars = "*" if parameter.kind is parameter.VAR_POSITIONAL else "**"
            raise TypeError(
                f"its parameter `{stars}{parameter.name}` is variadic, and every argument "
                "the tool receives must be judged by name. Declare each parameter by name."
            )
        if parameter.kind is parameter.KEYWORD_ONLY:
            keyword.append(parameter.name)
        else:
            positional.append(parameter.name)
    return positional, keyword


def _unprepared_parameters(schema: FuncSchema) -> Optional[str]:
    """Why the SDK will not prepare this tool's arguments before the tool runs, or None if
    it will. The rule is the SDK's own (``can_prepare_without_user_code``): it prepares a
    parameter model only when validating it runs none of the tool's code.

    A parameter of a type the SDK does prepare is still not prepared when its default is
    not JSON data (``tags: list[str] = ()``, ``timeout: float = math.inf``) or comes from a
    factory other than ``list`` or ``dict``; those two causes are named for what they are."""
    model = schema.params_pydantic_model
    if can_prepare_without_user_code(model):
        return None
    names: list[str] = []
    defaults: list[str] = []
    factories: list[str] = []
    try:
        fields = model.__pydantic_core_schema__["schema"]["fields"]
        for name, field in fields.items():
            inner = field["schema"]
            if _is_data_only_schema(inner):
                continue
            if inner["type"] != "default" or not _is_data_only_schema(inner["schema"]):
                names.append(name)
            elif inner.get("default_factory") is not None:
                factories.append(name)
            else:
                defaults.append(name)
    except Exception:  # noqa: BLE001, the refusal stands without the parameter's name
        names, defaults, factories = [], [], []
    if not names and factories:
        return (
            f"parameter {_quoted(factories)} has a default factory other than `list` or "
            "`dict`, which is code of the tool's own that the SDK does not run before the "
            "tool does, so no call could be judged. Use `Field(default_factory=list)`, "
            "`Field(default_factory=dict)` or a plain default, and build the value inside "
            "the tool body."
        )
    if not names and defaults:
        return (
            f"parameter {_quoted(defaults)} has a default that is not plain JSON data (a "
            "tuple, a set, a non-finite number, an object), which the SDK does not "
            "prepare, so no call could be judged. Give it a JSON default (None, a string, a "
            "finite number, a boolean, a list or a dict of them) and build the other value "
            "inside the tool body."
        )
    which = f"parameter {_quoted(names)}" if names else "a parameter"
    return (
        f"{which} has a type or a validation the SDK prepares only by running the tool's own "
        "code (an Enum, a model, a dataclass, a TypedDict, a date, a tuple, a set, a Sequence "
        "or an Iterable, bytes, a pattern, a custom validator), so no call could be judged. "
        "Declare data-only parameters (str, int, float, bool, None, Literal, and lists and "
        "dicts of them) and resolve application types inside the tool body."
    )


def _quoted(names: list[str]) -> str:
    return ", ".join(f"`{name}`" for name in names)


def _string_literals(keys: dict) -> bool:
    """A key type that is a ``Literal`` of strings only: the SDK keeps such a key as the
    string it was sent as, and refuses any other."""
    expected = keys.get("expected")
    return (
        keys.get("type") == "literal"
        and type(expected) is list
        and bool(expected)
        and all(type(value) is str for value in expected)
    )


def _other_keys(node: Any) -> bool:
    """Whether a parameter's core schema holds a mapping whose key type is neither ``str``
    nor a ``Literal`` of strings."""
    if type(node) is dict:
        keys = node.get("keys_schema")
        if node.get("type") == "dict" and type(keys) is dict:
            if keys.get("type") not in ("str", "any") and not _string_literals(keys):
                return True
        return any(_other_keys(item) for item in node.values())
    if type(node) is list:
        return any(_other_keys(item) for item in node)
    return False


def _mapping_keys_refusal(schema: FuncSchema) -> Optional[str]:
    """Why a mapping parameter could not be judged, or None.

    The SDK accepts ``dict[int, str]`` and converts the keys of the JSON object it is sent
    (``{"1": "a"}`` reaches the function as ``{1: "a"}``). The gate judges the prepared
    arguments as JSON data, whose keys are strings, so every call that gives such a mapping
    a member would be refused; the tool is refused when it is gated."""
    try:
        fields = schema.params_pydantic_model.__pydantic_core_schema__["schema"]["fields"]
        names = [name for name, field in fields.items() if _other_keys(field["schema"])]
    except Exception:  # noqa: BLE001, a parameter model the gate cannot read is refused
        return "its parameter model cannot be read"
    if not names:
        return None
    return (
        f"parameter {_quoted(names)} is a mapping whose key type is not `str` (or a `Literal` "
        "of strings). The SDK "
        "would convert the keys of the JSON object the model sends, and the gate judges "
        "arguments as JSON data, whose keys are strings, so no call that gives it a member "
        "could be judged. Declare `dict[str, ...]` and convert the keys inside the tool body."
    )


#: The object ``functools.lru_cache`` returns, which the kernel's walk reads through.
_LRU_WRAPPER = type(functools.lru_cache(maxsize=None)(lambda: None))

#: How every reason that says what to write ends. The advice of each reason removes the
#: cause; this says why calling the refused function from a plain function does not.
_BODY_NOT_READ = (
    "The gate reads a plain function's parameters and never its body, so a plain function "
    "that calls the refused one is gated with the cause still inside it."
)

#: For a reason that quotes the kernel's text: that text ends with the kernel's own advice,
#: which in ``recusal`` 0.17.0 can be to register a plain function that calls the refused
#: one; this says which plain function is meant. The kernel's advice is under revision on
#: its own branch (kernel-0.17.1); once the floor is a kernel whose advice removes the
#: cause, this note and the tests that hold it can go.
_KERNEL_ADVICE = (
    "Where that report advises a plain function, write one that does the work itself and "
    "does not call the refused one."
)

_SINGLEDISPATCH = (
    "it runs a `functools.singledispatch` function: the implementation that runs is chosen "
    "at call time by the type of an argument, one can be registered at any moment, and "
    "each has its own parameters and defaults, so the gate cannot tie a call to the "
    "function that will run. Register one plain function as the tool, with the code of "
    "each implementation in its own body under a branch on the type of the argument "
    "(`if isinstance(...)`), and no single-dispatch function. " + _BODY_NOT_READ
)

_DISPATCH_NOT_READ = (
    "the gate could not read a callable the call reaches ({}), so it cannot tell whether "
    "that is one it refuses (a `functools.singledispatch` function, a "
    "`functools.partialmethod`, a weak proxy). Register the function the tool is to run "
    "itself, without the callable that could not be read on the way to it. " + _BODY_NOT_READ
)

_PARTIALMETHOD_REFUSED = (
    "it runs a `functools.partialmethod`: the call goes to the callable the partialmethod "
    "holds, and the tool's parameters are read from that callable without the gate "
    "reading what is under it, so a wrapper or a single-dispatch function there could run "
    "with a value no policy judged. Write the method out in place of the partialmethod, "
    "with the values it binds as defaults of its own parameters (`def export(self, sql, "
    "force=False)` in place of `export = functools.partialmethod(run, force=False)`), "
    "and register that. " + _BODY_NOT_READ
)

_WEAK_PROXY = (
    "a callable the call reaches is a `weakref.proxy`: it shows the attributes of the "
    "object it stands for and is not that object, so the gate cannot tell which callables "
    "a call to it runs. Put the object the proxy stands for in its place, so that nothing "
    "on the way to the function is a proxy. " + _BODY_NOT_READ
)

#: For the kernel walk's ``OpenDispatch`` (``recusal`` 0.17.0): a single-dispatch function,
#: or the callable a ``singledispatchmethod`` hands out, whose registry is exactly its base.
#: This gate refuses every dispatcher its own walk reaches with ``_SINGLEDISPATCH``, and
#: its walk stands wherever the kernel's does, so no tool is known that gets this reason;
#: it is there for one only the kernel's walk would find. A dispatcher a decorator HOLDS
#: is refused by the kernel as a ``HiddenWrapper``, with that reason. True of every one:
#: it runs its base now, and one that nothing is ever registered on runs correctly.
_OPEN_DISPATCH = (
    "it runs a single-dispatch function (`functools.singledispatch`, or the callable a "
    "`singledispatchmethod` hands out) that has nothing registered but its base. It runs "
    "that base now, but an implementation can be registered on it at any moment, also "
    "after the gate checked a call and before it runs (by another call of the same model "
    "turn, or by a wrapper on its first use), and that implementation would then run for "
    "the judged call, with any parameter of its own at a default no policy judged. A tool "
    "refused for this runs correctly as long as nothing is ever registered on it. Where "
    "nothing is to be registered, register the base function itself; otherwise register "
    "one plain function as the tool, with the code of each implementation in its own body "
    "under a branch on the type of the argument. " + _BODY_NOT_READ
)

#: For the kernel walk's ``ProxiedCallable`` that is not a ``WeakProxy`` (``recusal``
#: 0.17.0): a callable on the way answers for ``__wrapped__`` without holding one, or shows
#: another class than its own. The kernel's own text goes in the braces, whole: it says
#: which of the two was found, names no value, and ends with the kernel's advice. That
#: advice can be to register a plain function that calls it, which would hide the cause:
#: ``_KERNEL_ADVICE`` says which plain function is meant. A tool refused for this may run
#: correctly.
_PROXIED_CALLABLE = (
    "a callable on the way to the function it runs answers for another object (it answers "
    "for `__wrapped__` without holding one, as a proxy or a class-based decorator that "
    "hands every attribute on with `__getattr__` does, or it shows another class than its "
    "own), so the gate cannot tell what a call to it runs. A tool refused for this may run "
    "correctly. The kernel's walk reports: {}. A class-based decorator that keeps the "
    "function in an attribute and hands the other attributes on with `__getattr__` is read "
    "once it holds `__wrapped__` itself: call `functools.update_wrapper(self, fn)` in its "
    "`__init__`. For any other such object, register the function it calls itself. "
    + _KERNEL_ADVICE
    + " "
    + _BODY_NOT_READ
)

#: For the kernel walk's ``HiddenWrapper``. The kernel's own text goes in the braces. It
#: says what was found: a parameter of the held wrapper that the tool does not declare, a
#: parameter the tool does declare that the held wrapper takes itself or takes in another
#: place, or only that the way from the held wrapper to the function could not be judged
#: (it passes a single-dispatch function, a weak proxy of a helper, a cache named again
#: over something other than a plain function, or does not end), with no parameter named.
#: It never holds a value, and it ends with the kernel's advice (``_KERNEL_ADVICE``). A
#: tool refused for the last cause may run correctly.
_HIDDEN_WRAPPER = (
    "a decorator on the function it runs holds a second wrapper of it that the gate does "
    "not follow (in its closure, a global it names or an attribute of the decorator "
    "object, and not reached through `__wrapped__`): {}. Remove the cause from the "
    "wrapper the decorator holds: give it no parameter of its own (pass such a value "
    "explicitly in its body), give it the tool's parameters in the tool's order or as "
    "keyword-only ones, and hold no single-dispatch function, weak proxy or cache named "
    "again on its way to the function; or register the function itself, without the "
    "decorator. " + _KERNEL_ADVICE + " " + _BODY_NOT_READ
)

#: For the kernel walk's ``UnreadCache``, which is a ``HiddenWrapper`` and is asked for
#: first. Raised where what the cache refers to is not laid out as the kernel knows it.
_UNREAD_CACHE = (
    "a `functools.lru_cache` wrapper on the way to the function it runs cannot be read: "
    "which function the cache calls could not be told from what the cache refers to, and "
    "that need not be the function it names as `__wrapped__`, so a wrapper with a "
    "parameter of its own could run under the cache with a value no policy judged. "
    "Register the function itself, without the cache. " + _BODY_NOT_READ
)

#: For the kernel walk's ``BoundArgumentKept``. The framing is this gate's and holds for
#: every one: which parameter is left open, and whether it then takes a keyword, runs at
#: a default or makes the call fail, is not said here. The kernel's own text goes in the
#: braces, whole: it names one parameter, or two where a wrapper took the bound argument
#: under a name, and no value, and it ends with the kernel's advice
#: (``_KERNEL_ADVICE``). A wrapper of this signature that runs correctly is refused too.
_BOUND_ARGUMENT_KEPT = (
    "a wrapper on the way to the function it runs is handed a bound argument (the object "
    "of a method, or a positional argument of a `functools.partial`), and the gate cannot "
    "tell from the wrapper's signature whether it passes that argument on, keeps it or "
    "passes a value of its own in its place, so it cannot tell which parameter of the "
    "function each judged value fills. A tool refused for this may run correctly. The "
    "kernel's walk reports: {}. Register the function itself; or write the method without "
    "`functools.wraps`, naming the parameters the tool takes in its own signature, and pass "
    "them on to the function by keyword. " + _KERNEL_ADVICE + " " + _BODY_NOT_READ
)

#: For a method whose function is a wrapper of a function that was not written to take the
#: object (its first parameter is not named ``self`` or ``cls``). This SDK reads the tool's
#: parameters with ``inspect.signature``, which leaves that first parameter out as the
#: object's, and calls the tool with every declared parameter by position
#: (``FuncSchema.to_call_args``). The wrapper's signature does not show whether it hands
#: the object on, and where it does not, every value runs one place off: the first judged
#: value fills the first parameter. ``recusal`` 0.17.0 refuses such a facade only where
#: the parameter the object would fill has a default (``BoundArgumentKept``), so the gate
#: refuses the rest itself (:func:`_kept_receiver`). Names a parameter, never a value.
_KEPT_RECEIVER = (
    "a wrapper on the way to the function it runs (it names `__wrapped__`) is handed an "
    "argument bound on the way (the object of a method, or a positional argument of a "
    "`functools.partial`) in the place of `{name}`, a parameter of the function it wraps "
    "that is not named as an object's (`self` or `cls`). The SDK leaves that parameter "
    "out of the tool's parameters and passes the judged values on by position, so where "
    "the wrapper does not hand the bound argument on, every value runs one parameter off "
    "(the first as `{name}`). Whether it does cannot be read from the wrapper's signature, "
    "so the tool is refused, and one that does hand it on is refused too. Register the "
    "function itself, not the method or the partial that wraps it. " + _BODY_NOT_READ
)

#: For the kernel walk's ``DeclaredSignature``: a ``__signature__``, a
#: ``__text_signature__`` that is not the interpreter's own, or a code object shown by an
#: object called through a ``__call__`` written in Python.
#: True of every one: nothing is said of whether this Python reads the declaration, or of
#: whether it differs from the parameters the callable binds.
_DECLARED_SIGNATURE = (
    "the function it runs, or a callable on the way to it, carries a declared signature (a "
    "`__signature__` attribute, whatever its value, a `__text_signature__`, or a code "
    "object a callable object shows). `inspect.signature` can read such a declaration in "
    "place of the parameters Python binds, and which ones it reads differs between Python "
    "versions, so the kernel's walk does not rely on any: it refuses each one, also where "
    "this Python would not read it and where it matches the real parameters. A tool "
    "refused for this may run correctly. Remove the declaration (delete the "
    "`__signature__` or the `__text_signature__`, or the `__code__` the object shows), so "
    "that the parameters Python binds are what is read. " + _BODY_NOT_READ
)

#: The kernel walk's own message for a stack it stops reading.
_WALK_LIMIT = "too many wrappers to inspect"

_TOO_MANY_WRAPPERS = (
    "the function it runs is under more wrappers than the gate reads (a stack of more than "
    "31 wrappers, or more than eight `functools.lru_cache` layers one on another), so the "
    "functions under them cannot be read. Take wrappers off the function until the gate "
    "reads them all, or register the function itself. " + _BODY_NOT_READ
)

#: For a callable on the way that is an object whose class defines ``__get__`` and for
#: which ``inspect.signature`` finds no signature (:func:`_opaque_decorator`).
_OPAQUE_DECORATOR = (
    "a callable on the way to the function cannot be inspected: it is an object whose "
    "class defines `__get__`, for which Python shows no signature (a class-based decorator "
    "that is also a descriptor, such as `cachetools.cached`, or a `staticmethod` object "
    "that something other than a class holds), so the functions under it cannot be read. "
    "Register the function itself, without that decorator, or use one the gate reads (a "
    "function decorator written with `functools.wraps`, or a class-based decorator with no "
    "`__get__` that calls `functools.update_wrapper(self, fn)`); hold a function where a "
    "`staticmethod` object stands. " + _BODY_NOT_READ
)


#: A ``__call__`` implemented in C (a slot wrapper or a bound one): nothing further in.
_C_CALL = (types.WrapperDescriptorType, types.MethodWrapperType)

#: How many callables :func:`_reached_callables` reads before it gives up. Far more than
#: the kernel's walk accepts, so a tool this stops at is one that walk refuses as well.
_MAX_REACHED = 512


#: ``functools.partialmethod`` as it was when this module was imported.
_PARTIALMETHOD: Any = getattr(functools, "partialmethod", None)

#: The names under which the function ``functools`` hands out for a partialmethod carries
#: it: ``_partialmethod`` up to Python 3.12, ``__partialmethod__`` from 3.13. The third is
#: asked as well, so a spelling without underscores is not missed.
_PARTIALMETHOD_NAMES = ("_partialmethod", "__partialmethod__", "partialmethod")


def _is_partialmethod(value: Any) -> bool:
    """Whether ``value`` is a ``functools.partialmethod``: its type is that class or a
    subclass of it, as the class was at import or as ``functools`` has it now, or a class
    of that name (one built by another copy of ``functools``). Asked of the type alone."""
    for known in (_PARTIALMETHOD, getattr(functools, "partialmethod", None)):
        if isinstance(known, type) and issubclass(type(value), known):
            return True
    return type(value).__name__ == "partialmethod"


def _held_partialmethod(link: Any) -> Any:
    """The ``functools.partialmethod`` that ``link`` is, or carries under one of
    ``_PARTIALMETHOD_NAMES``, or None. Read with ``inspect.getattr_static``.

    ``inspect.signature`` reads the first two names as attributes, so a class member of
    either name counts. The third is no name Python reads: it is asked so that a mark
    under a spelling without underscores is not missed, and only where the object itself
    holds it (a mark ``functools`` or ``functools.update_wrapper`` puts there). A class
    member of that name is an ordinary attribute of the class, and a callable object whose
    class has one is not refused for it."""
    if _is_partialmethod(link):
        return link
    for name in _PARTIALMETHOD_NAMES:
        held = inspect.getattr_static(link, name, None)
        if name == "partialmethod" and not issubclass(type(link), type):
            member = inspect.getattr_static(type(link), name, None)
            if member is not None and held is member:
                continue
        if _is_partialmethod(held):
            return held
    return None


def _steps(link: Any) -> list[Any]:
    """The callables ``link`` may hand a call to. Five steps the kernel's walk takes
    (``recusal.invocation._next_callables``): a ``functools.partial``'s function, a bound
    method's function, what ``__wrapped__`` names (as the callable shows it, so a method's
    function's, a ``functools.lru_cache`` wrapper's and the function under a
    ``staticmethod`` or a ``classmethod`` are followed), the function a
    ``functools.lru_cache`` wrapper calls, and the ``__call__`` of its type as the class
    holds it, unless that is implemented in C, as a function's own is. And two steps
    that walk does not take: what a cache named again names (below), and the callable a
    ``functools.partialmethod`` holds, when ``link`` is one or carries one.

    The function a cache calls is not always the one it names: ``functools.wraps(f)``
    written above the cache, or ``functools.update_wrapper(cached, f)``, names ``f``
    while the cache calls the function it was built over. Both are followed here: what
    the cache names, which the kernel's walk does not follow from such a cache, and what
    the kernel's ``cache_target`` says it calls, which is the step that walk takes (0.2.2
    followed the name alone). So the questions asked of these callables (a
    single-dispatch function, a partialmethod, a weak proxy) are asked of what such a
    cache calls too. ``cache_target`` is asked only of an object whose own type is the
    cache's (a ``weakref.proxy`` of a cache is not one: it is refused as a proxy), and
    it raises ``UnreadCache`` where what the cache calls cannot be told, which
    :func:`_wrapper_refusal` turns into a refusal of its own. It calls nothing the cache
    holds.

    ``__wrapped__`` is read with ``getattr``, as Python and the kernel's walk read it, so
    a ``__wrapped__`` property or a ``__getattr__`` of the callable runs here. ``__call__``
    and the partialmethod are read with ``inspect.getattr_static``."""
    found: list[Any] = []
    if isinstance(link, functools.partial):
        found.append(link.func)
    if inspect.ismethod(link):
        found.append(link.__func__)
    wrapped = getattr(link, "__wrapped__", None)
    if wrapped is not None:
        found.append(wrapped)
    if type(link) is _LRU_WRAPPER:
        # what the cache calls, which is what it names unless it was named again
        found.append(_cache_target(link))
    call = inspect.getattr_static(type(link), "__call__", None)
    if call is not None and not isinstance(call, _C_CALL):
        found.append(call)
    held = _held_partialmethod(link)
    if held is not None:
        found.append(inspect.getattr_static(held, "func", None))
    return found


def _reached_callables(func: Any) -> tuple[list[Any], bool]:
    """``func`` and every callable :func:`_steps` leads to from it, each once, and
    whether that is all of them (False when the walk stopped at ``_MAX_REACHED``)."""
    reached: list[Any] = []
    pending = [func]
    while pending:
        link = pending.pop()
        if link is None or any(link is seen for seen in reached):
            continue
        if len(reached) >= _MAX_REACHED:
            return reached, False
        reached.append(link)
        pending.extend(_steps(link))
    return reached, True


def _singledispatch_wrapper_code() -> Any:
    """The code object of the function ``functools.singledispatch`` returns, as
    ``functools`` stands when this is called: the one ``wrapper`` nested in it. None when
    it is not found by that name, or when what stands at ``functools.singledispatch`` is
    not a plain function (an instrumented or replaced one). Never raises."""
    try:
        code = functools.singledispatch.__code__
        if type(code) is not types.CodeType:
            return None
        found = [
            const
            for const in code.co_consts
            if type(const) is types.CodeType and const.co_name == "wrapper"
        ]
    except Exception:  # noqa: BLE001, whatever stands there, importing this module goes on
        return None
    return found[0] if len(found) == 1 else None


#: That code object as it was when this module was imported. One of three ways a
#: dispatcher is known (:func:`_dispatch_function`), and never the only one.
_SINGLEDISPATCH_WRAPPER = _singledispatch_wrapper_code()


def _closure_value(func: Any, name: str) -> Any:
    """What a plain function's closure holds under ``name``, or None."""
    try:
        return func.__closure__[func.__code__.co_freevars.index(name)].cell_contents
    except ValueError:  # a name the closure does not hold, or an empty cell
        return None


#: What the ``dispatch`` function inside a ``functools.singledispatch`` dispatcher closes
#: over beside its ``registry``, on every Python from 3.10 to 3.15.
_FUNCTOOLS_DISPATCH_STATE = frozenset({"dispatch_cache", "cache_token"})


def _dispatch_function(link: Any) -> bool:
    """Whether ``link`` is the function ``functools.singledispatch`` returns, with or
    without an implementation registered. Yes if any one of three tests says so:

    - its code is the ``wrapper`` nested in ``functools.singledispatch``, as read when
      this module was imported or as ``functools`` stands now (after a reload);
    - it is a plain function whose closure holds a ``dispatch`` function whose closure
      holds a ``registry`` dict beside a ``dispatch_cache`` and a ``cache_token``, which
      is how ``functools`` builds one (Python 3.10 to 3.15), whichever copy of
      ``functools`` built it. A ``dispatch`` function over a ``registry`` dict alone, as
      an event bus of your own might have, is not one;
    - it has a ``registry`` that is exactly a ``types.MappingProxyType`` beside a callable
      ``dispatch`` and a callable ``register``. A ``functools.wraps`` wrapper over a
      dispatcher carries a copy of those and is taken for one, which is the same answer:
      it leads to the dispatcher.

    The attributes are read with ``inspect.getattr_static`` and the closure from its
    cells, so this test runs no property and no ``__getattr__`` of the object. A service
    object with a ``dispatch`` method and a ``registry`` that is not a mapping proxy is
    not one."""
    if type(link) is types.FunctionType:
        code = link.__code__
        if any(
            code is known for known in (_SINGLEDISPATCH_WRAPPER, _singledispatch_wrapper_code())
        ):
            return True
        dispatch = _closure_value(link, "dispatch")
        if type(dispatch) is types.FunctionType:
            if type(_closure_value(dispatch, "registry")) is dict and (
                _FUNCTOOLS_DISPATCH_STATE <= set(dispatch.__code__.co_freevars)
            ):
                return True
    return (
        type(inspect.getattr_static(link, "registry", None)) is types.MappingProxyType
        and callable(inspect.getattr_static(link, "dispatch", None))
        and callable(inspect.getattr_static(link, "register", None))
    )


#: ``functools.singledispatchmethod`` as it was when this module was imported.
_SINGLEDISPATCHMETHOD: Any = getattr(functools, "singledispatchmethod", None)


def _dispatch_method_descriptor(value: Any) -> bool:
    """Whether ``value`` is a ``functools.singledispatchmethod``: an instance of that
    class as it was at import or as ``functools`` has it now, or an object of a class of
    that name whose ``dispatcher`` is a dispatch function (:func:`_dispatch_function`)."""
    for known in (_SINGLEDISPATCHMETHOD, getattr(functools, "singledispatchmethod", None)):
        if isinstance(known, type) and isinstance(value, known):
            return True
    return type(value).__name__ == "singledispatchmethod" and _dispatch_function(
        inspect.getattr_static(value, "dispatcher", None)
    )


def _dispatch_method(link: Any) -> bool:
    """Whether ``link`` is a ``functools.singledispatchmethod`` or the callable one hands
    out when it is read from an object or a class: up to Python 3.13 a function that
    carries the descriptor's bound ``register``, from 3.14 an object that holds the
    descriptor as ``_unbound``. The attributes are read with ``inspect.getattr_static``,
    so no property and no ``__getattr__`` of the object runs; the ``isinstance`` tests
    read the object's type and, when that is not the class, its ``__class__``."""
    if _dispatch_method_descriptor(link):
        return True
    register = inspect.getattr_static(link, "register", None)
    if inspect.ismethod(register) and _dispatch_method_descriptor(register.__self__):
        return True
    return _dispatch_method_descriptor(inspect.getattr_static(link, "_unbound", None))


def _is_dispatcher(link: Any) -> bool:
    """A callable that picks the code it runs at call time by the type of an argument: a
    ``functools.singledispatch`` function or a ``singledispatchmethod``, registered or
    not."""
    return _dispatch_function(link) or _dispatch_method(link)


def _opaque_decorator(link: Any) -> bool:
    """An object ``inspect`` reads as a built-in and finds no signature for: an instance
    of a class with ``__get__``, as ``cachetools.cached`` returns, and a ``staticmethod``
    object reached as itself (``functools.partial(staticmethod(f))``)."""
    if not inspect.ismethoddescriptor(link) or isinstance(link, _LRU_WRAPPER):
        return False
    try:
        inspect.signature(link, follow_wrapped=False)
    except ValueError:
        return True
    except TypeError:
        return False
    return False


#: The names the first parameter of a method's function goes by, which mark it as the
#: object's place. Python binds the object there whatever the name; these are the evidence
#: that the function was written to take it.
_RECEIVER_NAMES = ("self", "cls")

_POSITIONAL = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)


def _kept_receiver(reached: list[Any]) -> Optional[str]:
    """Why a method among the callables the call reaches may hand the judged values on one
    place off, or None.

    For a bound method, ``inspect.signature`` reads the function its function wraps and
    leaves that function's first parameter out, as the object's place; the SDK then calls
    the method with every declared parameter by position. Where the method's function is
    a wrapper (it names ``__wrapped__``) of a function whose first parameter is named
    otherwise than ``self`` or ``cls``, that parameter was not written as the object's,
    and whether the wrapper hands the object on to it, or keeps it and passes the judged
    values on so that the first of them fills it, cannot be read from the wrapper's
    signature. Refused, with ``_KEPT_RECEIVER``. A method of an ordinary class that a
    decorator wraps (``def run(self, sql)``) is read as before.

    The same holds for a ``functools.partial`` that binds positional arguments to such a
    wrapper: ``inspect.signature`` leaves out the parameters they fill, and the SDK's
    values fill them where the wrapper drops the bound arguments. Refused where one of
    those parameters is named otherwise than ``self`` or ``cls``. Names a parameter, never
    a value."""
    for link in reached:
        if inspect.ismethod(link):
            function, bound = link.__func__, 1
        elif isinstance(link, functools.partial) and link.args:
            function, bound = link.func, len(link.args)
        else:
            continue
        if getattr(function, "__wrapped__", None) is None:
            continue  # not a wrapper: Python binds the arguments to its own parameters
        # the signature ``inspect.signature`` takes the bound places from: that of the
        # function, read through its ``__wrapped__`` chain, as it reads it for the tool
        parameters = list(inspect.signature(function).parameters.values())
        for parameter in parameters[:bound]:
            if parameter.kind not in _POSITIONAL:
                break
            if parameter.name not in _RECEIVER_NAMES:
                return _KEPT_RECEIVER.format(name=parameter.name)
    return None


def _unread_reason(exc: Exception, reached: list[Any]) -> str:
    """The reason a tool is refused with when the kernel's walk
    (``recusal.invocation.reached_parameters``) raised ``exc`` for the function it runs.

    Each refusal the walk makes by a class of its own is reported as what it is, with
    what to do. A single-dispatch function and a ``functools.partialmethod`` get the
    reasons this gate gives the ones it finds itself. The text of a ``HiddenWrapper`` and
    of a ``BoundArgumentKept`` is the kernel's and is passed on: it is built from
    parameter names, and from no value. An ``UnreadCache`` is a ``HiddenWrapper``, so it
    is asked for first. A single-dispatch function with nothing registered
    (``OpenDispatch``, which is not a ``DispatchedAtCallTime``) and a callable that
    answers for another object (``ProxiedCallable``) have reasons of their own; a weak
    proxy (``WeakProxy``, a ``ProxiedCallable``) gets the reason this gate gives the ones
    it finds itself, and is asked for first. Anything else the walk raises names only its
    class: its text is not the kernel's to vouch for."""
    if isinstance(exc, _DispatchedAtCallTime):
        return _SINGLEDISPATCH
    if isinstance(exc, _OpenDispatch):
        return _OPEN_DISPATCH
    if isinstance(exc, _WeakProxy):
        return _WEAK_PROXY
    if isinstance(exc, _ProxiedCallable):
        return _PROXIED_CALLABLE.format(exc)
    if isinstance(exc, _PartialMethodSignature):
        return _PARTIALMETHOD_REFUSED
    if isinstance(exc, _DeclaredSignature):
        return _DECLARED_SIGNATURE
    if isinstance(exc, _BoundArgumentKept):
        return _BOUND_ARGUMENT_KEPT.format(exc)
    if isinstance(exc, _UnreadCache):
        return _UNREAD_CACHE
    if isinstance(exc, _HiddenWrapper):
        return _HIDDEN_WRAPPER.format(exc)
    if type(exc) is ValueError and any(_opaque_decorator(link) for link in reached):
        return _OPAQUE_DECORATOR
    if type(exc) is ValueError and str(exc) == _WALK_LIMIT:
        return _TOO_MANY_WRAPPERS
    return f"the function it runs cannot be inspected ({type(exc).__name__})"


def _wrapper_refusal(tool: FunctionTool, schema: FuncSchema) -> Optional[str]:
    """Why a callable on the way to the tool's code could run with a value no policy
    judged, or None.

    The SDK reads a ``functools.wraps`` wrapper as the function it wraps, prepares the
    arguments for that function's parameters and passes them to the wrapper, by position
    where it can. So every callable the call reaches (the kernel's ``reached_parameters``
    walk) must either take exactly the parameters the SDK prepared for, or forward
    ``*args`` and ``**kwargs`` and name none of its own. A wrapper that adds a parameter
    runs it at a default that is in no schema; one that takes a parameter of the function
    leaves the function at its own default; one that reorders them swaps the values.

    The walk reads the callable, what ``__wrapped__`` leads to, the function a
    ``functools.lru_cache`` wrapper calls, a bound method's function, a callable object's
    ``__call__`` and the callable a ``functools.partialmethod`` holds (:func:`_steps`).
    Three kinds of callable are refused wherever they stand among those:

    - a ``functools.singledispatch`` function or a ``singledispatchmethod``, with or
      without an implementation registered: which implementation runs is chosen at call
      time, and one can be registered between the moment a call is judged and the moment
      it runs (the SDK judges every call of a model turn before any tool body runs);
    - a ``weakref.proxy``: reading ``__wrapped__`` from it reads the attribute of the
      object it stands for, so the walk would step past that object, and a dispatcher
      behind a proxy would not be asked;
    - a callable that is or carries a ``functools.partialmethod``: ``inspect`` reads the
      parameters of the callable the partialmethod holds as the tool's own, and neither
      it nor the kernel's walk reads what is under that callable (the walk refuses one
      it finds, with ``PartialMethodSignature``). That is what
      reading a partialmethod from a class gives, and from an object when the callable
      it holds has no ``__get__``. Over a plain function read from an object it gives a
      ``functools.partial``, which the SDK refuses itself.

    The questions are asked of every callable :func:`_reached_callables` finds; a
    callable they cannot be asked of is refused, and a cache whose function cannot be
    told (``UnreadCache``, raised by the kernel's ``cache_target`` in :func:`_steps`) is
    refused with the reason the kernel's walk gets for it. A function a wrapper holds in
    a closure, a global or an attribute is not among them. The kernel's walk refuses the
    shapes of that its docstrings list (``HiddenWrapper``), and what it raises is turned
    into a reason by :func:`_unread_reason`."""
    get_callable = getattr(tool.on_invoke_tool, "_get_wrapped_callable", None)
    func = get_callable() if callable(get_callable) else None
    try:
        reached, complete = _reached_callables(func)
        dispatches = any(_is_dispatcher(link) for link in reached)
        proxied = any(type(link) in weakref.ProxyTypes for link in reached)
        partialmethod = any(_held_partialmethod(link) is not None for link in reached)
    except _UnreadCache:
        return _UNREAD_CACHE
    except Exception as exc:  # noqa: BLE001, a callable the gate cannot question is refused
        return _DISPATCH_NOT_READ.format(type(exc).__name__)
    if dispatches:
        return _SINGLEDISPATCH
    if proxied:
        return _WEAK_PROXY
    if partialmethod:
        return _PARTIALMETHOD_REFUSED
    if not complete:
        return _TOO_MANY_WRAPPERS
    try:
        if not callable(func):
            raise TypeError("the invoker does not expose the function it wraps")
        levels = _reached_parameters(func)
    except Exception as exc:  # noqa: BLE001, a callable the gate cannot read is refused
        return _unread_reason(exc, reached)
    try:
        kept = _kept_receiver(reached)
    except Exception as exc:  # noqa: BLE001, a callable the gate cannot question is refused
        return _DISPATCH_NOT_READ.format(type(exc).__name__)
    if kept is not None:
        return kept
    declared = [(p.name, p.kind) for p in schema.signature.parameters.values()]
    for level in levels:
        own = [(p.name, p.kind) for p in level.values()]
        named = [name for name, kind in own if kind not in _VARIADIC]
        if named and own != declared:
            known = {name for name, _ in declared}
            name = next((n for n in named if n not in known), named[0])
            return (
                f"a function the call reaches takes `{name}` in a way the tool does not "
                "declare (a wrapper that adds a parameter of its own, or takes or reorders "
                "the function's), so it could run with a value no policy judged. A wrapper "
                "must forward `*args, **kwargs` and name no parameter of its own."
            )
    return None


def _tool_refusal(tool: FunctionTool) -> Optional[str]:
    """Why no call to this function tool could be judged, or None if calls can be.

    ``gate_tools`` raises with this when the tool is gated, so a tool the gate would refuse
    on every call is found when the agent is built; the guardrail asks again on every call."""
    try:
        schema = _sdk_schema(tool)
        _judged_names(schema)
    except TypeError as exc:
        return str(exc)
    return (
        _unprepared_parameters(schema)
        or _mapping_keys_refusal(schema)
        or _wrapper_refusal(tool, schema)
    )


_AGENT_TOOL_NOT_READ = (
    "its invoker does not expose the adapter the SDK validates an agent tool's arguments "
    "with, so the gate cannot tell what the sub-agent would be given. Build the tool with "
    "`agent.as_tool(...)`."
)

_AGENT_TOOL_DATA_ONLY = (
    "The gate works out what the sub-agent is given only for a Pydantic model the SDK's "
    "own rule for data-only validation admits (`can_prepare_without_user_code`) and that "
    "is written out with no code of its own. Declare fields that are str, int, float, "
    "bool, None, Literal and lists of them, with JSON defaults, and build other types "
    "inside the sub-agent's own tools."
)


def _data_only(schema: Any) -> bool:
    """The SDK's rule for one core schema, for a node that may not be a schema at all."""
    return type(schema) is dict and "type" in schema and _is_data_only_schema(schema)


def _outside_cause(schema: Any) -> str:
    """What puts a field's core schema outside the SDK's data-only set, read the way the
    SDK's own rule reads it (``_is_data_only_schema``): under a default, an ``Optional``,
    a list, a mapping and a union, down to the part the rule stops at. A ``str`` and a
    ``list`` are in the set, so a string is named by its pattern and a list by what it
    holds; anything else by its pydantic core type."""
    kind = schema.get("type") if type(schema) is dict else None
    if kind in ("default", "nullable"):
        return _outside_cause(schema.get("schema"))
    if kind == "str" and "pattern" in schema:
        return "a string pattern"
    if kind == "list" and not _data_only(schema.get("items_schema", {"type": "any"})):
        return f"a list of {_outside_cause(schema.get('items_schema'))}"
    if kind == "dict":
        for key, part in (("keys_schema", "keys"), ("values_schema", "values")):
            if not _data_only(schema.get(key, {"type": "any"})):
                return f"a mapping with {part} of {_outside_cause(schema.get(key))}"
    if kind == "union":
        for choice in schema.get("choices") or ():
            if not _data_only(choice):
                return f"a union with {_outside_cause(choice)}"
    if kind == "literal":
        return "a `Literal` of values other than None, a bool, an int or a str"
    if type(kind) is str and kind.startswith("function-"):
        return f"a validator, pydantic core type `{kind}`"
    return f"pydantic core type `{kind}`" if type(kind) is str else "pydantic core type unknown"


def _agent_tool_parameters_reason(core: Any) -> str:
    """Why a ``parameters`` type is not one the gate works out the sub-agent's input for.
    A cause that lies in one field names the field and what puts it outside the
    data-only set (:func:`_outside_cause`); a cause on the type as a whole names that."""
    kind = core.get("type") if type(core) is dict else None
    if type(kind) is str and kind.startswith("function-"):
        return (
            "its `parameters` type has a validator of its own on the type as a whole (a "
            f"`model_validator`, for one; its pydantic core type is `{kind}`). "
            f"{_AGENT_TOOL_DATA_ONLY}"
        )
    if type(core) is not dict or core.get("type") != "model":
        kind = core.get("type") if type(core) is dict else "unknown"
        return (
            f"its `parameters` type is not a Pydantic model the gate reads: its pydantic "
            f"core type is `{kind}` (a dataclass is `dataclass`, a model that refers to "
            f"itself is `definitions`). {_AGENT_TOOL_DATA_ONLY}"
        )
    named: list[str] = []
    writes: list[str] = []
    try:
        inner = core["schema"]
        for name, field in inner["fields"].items():
            own = field["schema"]
            if not _is_data_only_schema(own):
                if own.get("type") == "default" and _is_data_only_schema(own["schema"]):
                    named.append(
                        f"`{name}` (a default that is not JSON data, or a default factory "
                        "other than `list` or `dict`)"
                    )
                else:
                    named.append(f"`{name}` ({_outside_cause(own)})")
            elif _runs_code_to_serialize(field):
                writes.append(f"`{name}`")
        computed = [f"`{each.get('property_name')}`" for each in inner.get("computed_fields") or ()]
    except Exception:  # noqa: BLE001, the refusal stands without the field's name
        named, writes, computed = [], [], []
    if named:
        return (
            f"its `parameters` model has a field outside the data-only set: "
            f"{', '.join(named)}. {_AGENT_TOOL_DATA_ONLY}"
        )
    if writes or computed:
        return (
            f"its `parameters` model writes {', '.join(writes + computed)} out through code "
            "of its own (a serializer, a computed field or `exclude_if`), which the SDK "
            f"runs again after the gate. {_AGENT_TOOL_DATA_ONLY}"
        )
    return (
        "its `parameters` model is not one the SDK's rule for data-only validation admits "
        "(a validator or a serializer on the model, a custom `__init__`, "
        "`model_post_init`, a private attribute, `defer_build`, a model that refers to "
        f"itself, a pydantic plugin). {_AGENT_TOOL_DATA_ONLY}"
    )


def _agent_tool_adapter(tool: FunctionTool) -> Any:
    """The adapter the SDK validates this agent tool's arguments with and writes the
    sub-agent's input from, or None for the default ``as_tool`` (one ``input`` string,
    which the SDK hands to the sub-agent as it is).

    Read from the SDK's own invocation function, which holds it, as :func:`_sdk_schema`
    reads a function tool's schema: one authority, never a second model built from the
    ``parameters`` type. Raises ``TypeError`` for an agent tool that carries none."""
    implementation: Any = getattr(tool.on_invoke_tool, "_invoke_tool_impl", None)
    try:
        adapter: Any = inspect.getclosurevars(implementation).nonlocals.get("params_adapter")
    except TypeError:
        adapter = None
    core = getattr(adapter, "core_schema", None)
    if type(core) is not dict or not all(
        callable(getattr(adapter, method, None)) for method in ("validate_python", "dump_python")
    ):
        raise TypeError(_AGENT_TOOL_NOT_READ)
    return None if core.get("cls") is AgentAsToolInput else adapter


def _runs_code_to_serialize(node: Any) -> bool:
    """Whether a core schema writes a value out through code of the model's own: a
    serializer on a field or on the model, a computed field, or a field left out by a
    callable (``Field(exclude_if=...)``)."""
    if type(node) is dict:
        if (
            "serialization" in node
            or node.get("computed_fields")
            or "serialization_exclude_if" in node
        ):
            return True
        return any(_runs_code_to_serialize(item) for item in node.values())
    if type(node) is list:
        return any(_runs_code_to_serialize(item) for item in node)
    return False


def _agent_tool_refusal(tool: FunctionTool) -> Optional[str]:
    """Why the input a sub-agent would be given could not be judged, or None if it can.

    With ``agent.as_tool(parameters=Model)`` the SDK validates the arguments against the
    model, which converts types, fills defaults and drops what the model does not name,
    and hands the sub-agent the result written back out as JSON. The gate judges that
    result, from the SDK's own adapter (:func:`_agent_tool_input`). The SDK then validates
    the arguments again itself, so the two agree only when validating and writing out run
    none of the model's own code; the rule for that is the SDK's
    (``can_prepare_without_user_code``, which reads a Pydantic model), with serializers
    and computed fields refused beside it. ``gate_tools`` raises with this when the tool
    is gated, and the guardrail asks again on every call."""
    try:
        adapter = _agent_tool_adapter(tool)
    except TypeError as exc:
        return str(exc)
    if adapter is None:
        return None
    core = adapter.core_schema
    try:
        data_only = (
            core.get("type") == "model"
            and can_prepare_without_user_code(core["cls"])
            and not _runs_code_to_serialize(core)
        )
    except Exception:  # noqa: BLE001, a parameters type the gate cannot read is refused
        data_only = False
    return None if data_only else _agent_tool_parameters_reason(core)


def _agent_tool_input(tool: FunctionTool, proposed: dict[str, Any]) -> dict[str, Any]:
    """What the sub-agent behind an agent tool is given for the arguments ``proposed``.

    For the default ``as_tool`` that is the arguments themselves. For
    ``as_tool(parameters=Model)`` it is what the SDK's own adapter makes of them, by the
    two calls the SDK makes: validated (types converted, defaults filled, other keys
    dropped) and written back out as JSON data. Returned as a detached copy. Raises
    ``ValueError`` when the arguments do not validate and ``TypeError`` when the result
    is not a JSON object."""
    adapter = _agent_tool_adapter(tool)
    if adapter is None:
        return proposed
    try:
        validated = adapter.validate_python(proposed)
    except Exception as exc:  # noqa: BLE001, pydantic's ValidationError, which is a ValueError
        raise ValueError("the arguments do not validate against its parameters") from exc
    try:
        given = _detached(adapter.dump_python(validated, mode="json"))
    except TypeError as exc:  # a non-finite number: text such as "nan" for a float field
        raise ValueError(f"the validated arguments are not JSON data: {exc}") from exc
    if type(given) is not dict:
        raise TypeError("the validated arguments are not a JSON object")
    return dict(given)


def _effective_arguments(schema: FuncSchema, prepared: Any) -> dict[str, Any]:
    """Name the invocation the SDK prepared: positional values bound to the parameters the
    SDK passes by position (its context parameter excluded, by its own ``takes_context``),
    keyword values as they are, so the policy and the audit fingerprint read exactly what
    the function will receive. The values are returned as a detached copy of JSON data.

    Raises ``TypeError`` when the prepared values are not one for each parameter, since the
    names would then be a guess, and when a value is not JSON data (a non-finite number)."""
    positional, keyword = _judged_names(schema)
    if len(prepared.args) != len(positional) or set(prepared.kwargs) != set(keyword):
        raise TypeError("the prepared values are not one for each of the tool's parameters")
    effective = dict(zip(positional, prepared.args))
    effective.update(prepared.kwargs)
    detached: dict[str, Any] = _detached(effective)
    return detached


def _uninspectable(prepared: Any) -> bool:
    """The SDK prepares nothing (empty, and not "unchanged") when the parameter model runs
    user code (a custom validator, a default factory) or the arguments do not validate."""
    return not prepared.args and not prepared.kwargs and not prepared.unchanged


def _proposed(
    tool_name: str, arguments: str, record: dict
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """The arguments the model proposed, as a JSON object: ``(arguments, None)``, or
    ``(None, reason)`` for text that is not one, and for arguments that are not text.

    The arguments are JSON text: exactly a ``str``, ``bytes`` or a ``bytearray``, which
    ``json.loads`` decodes. None stands for no arguments, as empty text does. Anything
    else (a ``dict`` that looks decoded already, a list, a number, a bool) is refused
    and not read: what would be executed for it is not known here, and ``json.loads``
    raises ``TypeError`` for it, which would leave this function as an exception and not
    as a refusal. That holds for an empty ``dict`` or list, ``0`` and ``False`` too,
    which were read as no arguments. The type is asked exactly, and the value is never
    asked whether it is true: an instance of a subclass of ``str`` or ``bytes`` can
    answer for its own truth and its own decoding (one that says it is false was read as
    no arguments), so it is refused like any other value. With a recording, such a
    refusal is bound to the name of the type the arguments arrived as, since they may
    not be data.

    A non-finite number (``NaN``, ``Infinity``, a literal too large for a float) is not
    JSON data: no policy can compare it and no record can fingerprint it, so it is refused
    here, before the policy runs, with or without a recording. So are arguments that hold
    a value more than ``_MAX_DEPTH`` levels down, before anything else reads them. Those
    refusals are bound to the argument text, as the one for malformed JSON is."""
    if arguments is None:
        text: Any = "{}"
    elif type(arguments) is str or type(arguments) is bytes or type(arguments) is bytearray:
        text = arguments if len(arguments) else "{}"
    else:
        kind = _type_name(arguments)
        return None, _unjudgeable(
            tool_name, _NOT_TEXT.format(kind), {"arguments_type": kind}, record
        )
    try:
        args = json.loads(text)
    except RecursionError:
        return None, _unjudgeable(tool_name, _TOO_DEEP, arguments, record)
    except ValueError:
        return None, _unjudgeable(tool_name, "arguments are not valid JSON", arguments, record)
    if _too_deep(args):
        return None, _unjudgeable(tool_name, _TOO_DEEP, arguments, record)
    try:
        args = _detached(args)
    except TypeError:
        return None, _unjudgeable(
            tool_name,
            "arguments hold a non-finite number, which is not JSON data",
            arguments,
            record,
        )
    if not isinstance(args, dict):
        return None, _unjudgeable(tool_name, "arguments are not a JSON object", args, record)
    return args, None


#: Given the call's tool context, the recording arguments ``decide`` takes for it
#: (``audit=``, ``surface=``, ``tool_use_id=``, ...). ``recusal_openai_agents.recording``
#: builds one.
Recording = Callable[[Any], dict]


def _unjudgeable(tool_name: str, detail: str, proposal: Any, record: dict) -> str:
    """The reason for a call the gate refuses before any policy runs.

    Without recording arguments it is the plain text. With them
    (the audited gate) the refusal goes through ``decide`` like every other,
    so it is one entry on the log, bound to the arguments as the model proposed them, and
    the reason the model reads is the one on the record.
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


def _refusal(tool_name: str, arguments: str, policy: Policy, **record: Any) -> str | None:
    """Judge a call by the JSON arguments as given: the reason it is refused, or None.

    This is the seam where the gate holds nothing but the argument text (the hosted MCP
    approval callback: the call is made on OpenAI's servers with these arguments, and what
    the MCP server there does with them, type conversion and defaults included, is not
    seen). Function tools go through :func:`_function_tool_refusal`, which judges the
    invocation the SDK prepared.
    ``record`` goes straight to ``decide``, which then writes the decision to an audit log.
    """
    args, refused = _proposed(tool_name, arguments, record)
    if args is None:
        return refused
    try:
        decision, reason = decide(tool_name, args, policy, safe_errors=True, **record)
    except Exception as exc:
        return f"Recusal failed closed (policy error): RECUSAL_POLICY_ERROR ({type(exc).__name__})"
    return reason if decision == "deny" else None


def _function_tool_refusal(
    tool_name: str,
    arguments: str,
    policy: Policy,
    tool: FunctionTool,
    ctx: Any = None,
    **record: Any,
) -> str | None:
    """Return the reason a function tool call is refused, or None to run it.

    The policy judges the invocation the SDK prepared from the arguments: validated against
    the tool's parameter model, defaults filled, types coerced, which is what the function
    receives, not the raw JSON the model wrote (``{"amount": "50"}`` reaches an ``int``
    parameter as ``50``; an omitted parameter reaches it as its default). When the call is
    allowed, that same prepared invocation is pinned on the tool context, so the SDK
    executes exactly what was judged, with no second validation in between.

    The SDK prepares nothing for a parameter model that runs user code (a custom validator,
    a default factory) or holds an application model (a nested ``BaseModel`` parameter),
    because preparing it would run the tool's code before the gate; such a tool is refused
    when it is gated, and its calls are refused here (:func:`_tool_refusal`), since the
    value it would execute cannot be judged. Validate inside the tool body and declare
    data-only parameters (scalars, lists and dicts of them). A prepared value that is not
    JSON data (the string ``"nan"`` read as a float) is refused the same way.

    A call to an agent tool (``agent.as_tool(...)``) has no prepared invocation: the policy
    judges the input the sub-agent is given (:func:`_agent_tool_input`). For the default
    ``as_tool`` that is the arguments as proposed; for ``as_tool(parameters=Model)`` it is
    the arguments as the SDK's adapter validates them and writes them back out, and
    arguments that do not validate are refused before the policy runs.

    ``record`` goes straight to ``decide``, so a gate that puts the decision on a log
    (``audited_agent``, ``audited_tools``) decides, and refuses, exactly as this one does. The
    entry for a judged call carries the fingerprint of the prepared invocation, named and
    projected to JSON data; a call refused before judgment is bound to the arguments as
    the model proposed them.
    """
    args, refused = _proposed(tool_name, arguments, record)
    if args is None:
        return refused
    if getattr(tool, "_is_agent_tool", False):
        # agent.as_tool(): the parent's policy judges the input the sub-agent is given, and
        # the sub-agent's own gate (which gate_tools requires) judges the calls it makes
        why = _agent_tool_refusal(tool)
        if why is not None:
            return _unjudgeable(tool_name, why, args, record)
        try:
            given = _agent_tool_input(tool, args)
        except Exception as exc:  # noqa: BLE001, an input the gate cannot work out is refused
            detail = str(exc) if type(exc) is ValueError else type(exc).__name__
            return _unjudgeable(
                tool_name,
                f"the input the sub-agent would be given cannot be judged ({detail})",
                args,
                record,
            )
        try:
            decision, reason = decide(tool_name, given, policy, safe_errors=True, **record)
        except Exception as exc:
            return (
                f"Recusal failed closed (policy error): RECUSAL_POLICY_ERROR ({type(exc).__name__})"
            )
        return reason if decision == "deny" else None
    why = _tool_refusal(tool)
    if why is not None:
        return _unjudgeable(tool_name, why, args, record)
    schema = _sdk_schema(tool)
    prepared = tool.on_invoke_tool.prepare_arguments(arguments, tool_name)  # type: ignore[attr-defined]
    if prepared is None or _uninspectable(prepared):
        if any(_judged_names(schema)):
            detail = "they do not validate against the tool's parameters"
        else:
            detail = "the tool takes no arguments and the call supplies some"
        return _unjudgeable(
            tool_name,
            f"the arguments the tool would execute cannot be judged: {detail}",
            args,
            record,
        )
    try:
        effective = _effective_arguments(schema, prepared)
    except Exception as exc:  # noqa: BLE001, an invocation the gate cannot name is refused
        detail = str(exc) if type(exc) is TypeError else type(exc).__name__
        return _unjudgeable(
            tool_name,
            f"the arguments the tool would execute cannot be judged ({detail})",
            args,
            record,
        )
    try:
        decision, reason = decide(tool_name, effective, policy, safe_errors=True, **record)
    except Exception as exc:
        return f"Recusal failed closed (policy error): RECUSAL_POLICY_ERROR ({type(exc).__name__})"
    if decision == "deny":
        return reason
    if ctx is not None:
        ctx._function_tool_arguments = prepared  # the SDK executes exactly what was judged
    return None


def _duplicate_names(tools: Any) -> list[str]:
    """Function-tool names that appear more than once, sorted; the SDK resolves such names
    last-wins at execution, so a guardrail could judge one tool while another runs."""
    names = [t.name for t in tools if isinstance(t, FunctionTool)]
    return sorted({n for n in names if names.count(n) > 1})


def _shared_name_reason(tools: Any) -> Optional[str]:
    """Why function tools that share a name cannot be gated, or None when no name is shared.

    Two tools under one name (and one ``tool_namespace``, or none) are resolved by the SDK
    last-wins. Two tools under one name in different namespaces are resolved by namespace
    and both can run, but a policy and the audit record are given the tool's own name
    without the namespace, so they could not tell the two apart."""
    duplicates = _duplicate_names(tools)
    if not duplicates:
        return None
    qualified = [
        (getattr(t, "_tool_namespace", None), t.name) for t in tools if isinstance(t, FunctionTool)
    ]
    if any(qualified.count(pair) > 1 for pair in qualified):
        return (
            f"function tools share a name {duplicates!r}; the SDK resolves a shared name "
            "last-wins at execution, so a guardrail could judge one tool while another runs. "
            "Give every function tool a unique name."
        )
    return (
        f"function tools in different namespaces share a name {duplicates!r}; a policy and "
        "the audit record are given a tool's own name without its namespace, so they could "
        "not tell the tools apart. Give every function tool a unique name."
    )


def _guardrail_for(policy: Policy, holder: list, recording: Optional[Recording] = None):
    """A tool input guardrail bound to ONE tool: the gated copy ``gate_tools`` puts into
    ``holder`` after building it. The guardrail judges and pins that object, never a tool
    rediscovered by name from the agent (``data.agent.tools`` is metadata, and with a shared
    name the SDK executes the last tool while a lookup would find the first). Unbound, it
    refuses every call. ``recording`` puts every decision, these refusals included, on an
    audit log; a ``recording`` that raises refuses the call."""

    def gate(data) -> ToolGuardrailFunctionOutput:
        ctx = data.context
        try:
            record = dict(recording(ctx)) if recording is not None else {}
        except Exception as exc:  # noqa: BLE001, a gate that cannot record must not run the call
            return ToolGuardrailFunctionOutput.reject_content(
                f"Recusal failed closed (policy error): RECUSAL_POLICY_ERROR ({type(exc).__name__})"
            )
        if not holder:
            return ToolGuardrailFunctionOutput.reject_content(
                _unjudgeable(
                    ctx.tool_name,
                    "this guardrail is bound to no tool; attach it with gate_tools or gated_agent",
                    ctx.tool_arguments,
                    record,
                )
            )
        tool = holder[0]
        if ctx.tool_name != tool.name:
            return ToolGuardrailFunctionOutput.reject_content(
                _unjudgeable(
                    ctx.tool_name,
                    f"this guardrail is bound to {tool.name!r}",
                    ctx.tool_arguments,
                    record,
                )
            )
        reason = _function_tool_refusal(
            ctx.tool_name, ctx.tool_arguments, policy, tool, ctx, **record
        )
        if reason is not None:
            return ToolGuardrailFunctionOutput.reject_content(reason)
        return ToolGuardrailFunctionOutput.allow()

    gate.recusal_gate = True  # type: ignore[attr-defined]
    # the decorator form: the SDK types the function and the name as separate overloads,
    # and builds the same ToolInputGuardrail either way
    return tool_input_guardrail(name=GUARDRAIL_NAME)(gate)


def recusal_guardrail(policy: Policy):
    """The guardrail as a standalone object, kept for the documented name: it is bound to no
    tool and therefore refuses every call. Attach guardrails with ``gate_tools`` or
    ``gated_agent``, which bind each one to the tool it protects."""
    return _guardrail_for(policy, [])


def _carries_gate(tool: FunctionTool) -> bool:
    return any(
        getattr(guardrail.guardrail_function, "recusal_gate", False)
        for guardrail in tool.tool_input_guardrails or ()
    )


def _ungateable_reason(agent: Any) -> str | None:
    """Why an agent can run a tool the guardrail never sees, or None if it cannot."""
    if type(agent) is not Agent:
        return (
            f"{type(agent).__name__} is not a plain Agent; a subclass can add tools at run "
            "time that the guardrail never sees"
        )
    if agent.mcp_servers:
        return "it has mcp_servers, whose tools the guardrail cannot see"
    if agent.handoffs:
        return "it has handoffs, which hand the conversation to another agent's tools"
    if agent.prompt is not None:
        return _PROMPT_REASON
    settings = _settings_reason(agent)
    if settings is not None:
        return f"its {settings}"
    shared = _shared_name_reason(agent.tools)
    if shared is not None:
        return f"its {shared}"
    return None


#: Why an agent with a ``prompt`` is not gated. With a prompt the SDK sends the request
#: under a prompt stored on OpenAI's servers, which can configure tools of its own.
_PROMPT_REASON = (
    "it has a prompt, and a prompt stored on OpenAI's servers can configure tools that are "
    "not in the agent's tools list, which no gate sees. Put the instructions and the tools "
    "on the agent itself."
)


#: Request fields that configure tools the provider runs, or a stored prompt that can.
_TOOL_REQUEST_KEYS = ("tools", "prompt", "plugins", "web_search_options", "mcp_servers")


def _settings_reason(agent: Any) -> Optional[str]:
    """Why an agent's ``model_settings`` could configure tools no gate sees, or None.

    The SDK merges ``extra_body`` into the body of every model request: a ``tools`` key
    there replaces the agent's own tool list in the request, and a stored ``prompt`` or
    another of these keys is sent beside it. ``extra_args`` are passed as arguments of the
    request call: a ``prompt`` there is sent with a Responses request, and a ``tools`` key
    makes the SDK's OpenAI models raise ``TypeError`` before anything is sent. All of
    these keys are refused in both. Only the agent's own
    settings are read: settings given to a run (``RunConfig(model_settings=...)``) are not
    known when the agent is gated."""
    settings = getattr(agent, "model_settings", None)
    for field in ("extra_body", "extra_args"):
        extra = getattr(settings, field, None)
        if not isinstance(extra, Mapping):
            continue
        found = [key for key in _TOOL_REQUEST_KEYS if key in extra]
        if found:
            return (
                f"model_settings.{field} sets {_quoted(found)}, which can put tools into the "
                "model request that are not in the agent's tools list, which no gate sees. "
                "Put the tools on the agent itself."
            )
    return None


def _tools_gated(tools: Any, agent_gated: Callable[[Any], bool]) -> bool:
    """True if every tool is a function tool that carries the Recusal guardrail. A sub-agent
    behind ``as_tool`` must also be gated itself, which ``agent_gated`` answers for it."""
    for tool in tools:
        if not isinstance(tool, FunctionTool) or not _carries_gate(tool):
            return False
        if getattr(tool, "_is_codex_tool", False):
            return False
        if getattr(tool, "_is_agent_tool", False):
            if not agent_gated(getattr(tool, "_agent_instance", None)):
                return False
    return True


def is_gated(agent: Any, _seen: set[int] | None = None) -> bool:
    """True if every tool this agent can run passes through a Recusal guardrail."""
    seen = set() if _seen is None else _seen
    if id(agent) in seen:
        return True
    seen.add(id(agent))
    if _ungateable_reason(agent) is not None:
        return False
    return _tools_gated(agent.tools, lambda sub: is_gated(sub, seen))


def gate_tools(
    tools: list[Any],
    policy: Policy,
    recording: Optional[Recording] = None,
    *,
    agent_gated: Optional[Callable[[Any], bool]] = None,
) -> list[FunctionTool]:
    """Return copies of the tools with the Recusal guardrail added; the originals are unchanged.

    Raises TypeError, naming the tool, for anything the guardrail cannot see and for a
    function tool no call to which could be judged (a parameter the SDK prepares only by
    running the tool's code, a variadic parameter, a wrapper with a parameter of its own,
    an agent tool whose ``parameters`` type runs code of its own).
    ``recording`` puts every decision of every guardrail it attaches on an audit log.
    ``agent_gated`` answers whether a sub-agent behind ``as_tool`` is itself gated; it is
    this package's ``is_gated`` unless a package that gates more seams passes its own.
    """
    shared = _shared_name_reason(tools)
    if shared is not None:
        raise TypeError(f"Recusal cannot gate these tools: {shared}")
    if agent_gated is None:
        agent_gated = is_gated
    gated = []
    for tool in tools:
        name = getattr(tool, "name", None) or "(unnamed)"
        if not isinstance(tool, FunctionTool):
            raise TypeError(
                f"Recusal cannot gate {type(tool).__name__} {name!r}: tool input guardrails "
                "run only on function tools"
            )
        if getattr(tool, "_is_codex_tool", False):
            raise TypeError(
                f"Recusal cannot gate the Codex tool {name!r}: it runs its own commands"
            )
        if not getattr(tool, "_is_agent_tool", False):
            why = _tool_refusal(tool)
            if why is not None:
                raise TypeError(f"Recusal cannot gate {name!r}: {why}")
        if getattr(tool, "_is_agent_tool", False):
            if not agent_gated(getattr(tool, "_agent_instance", None)):
                raise TypeError(
                    f"Recusal cannot gate agent tool {name!r}: its agent's own tool calls are "
                    "not gated. Use gated_agent(sub_agent, policy).as_tool(...)"
                )
            why = _agent_tool_refusal(tool)
            if why is not None:
                raise TypeError(f"Recusal cannot gate agent tool {name!r}: {why}")
        holder: list = []
        guardrail = _guardrail_for(policy, holder, recording)
        copy = dataclasses.replace(
            tool, tool_input_guardrails=[*(tool.tool_input_guardrails or ()), guardrail]
        )
        holder.append(copy)  # the guardrail judges the very object the agent will execute
        gated.append(copy)
    return gated


def gated_agent(agent: Agent, policy: Policy, recording: Optional[Recording] = None) -> Agent:
    """Return a copy of the agent whose every tool call is judged by Recusal.

    Raises ValueError if the agent can run anything the guardrail cannot see, and TypeError
    for a tool ``gate_tools`` refuses. The copy has its own ``tools``, ``handoffs`` and
    ``mcp_servers`` lists: ``Agent.clone`` alone would share the original's, and a handoff
    appended to the original afterwards would then be the gated copy's too.
    """
    reason = _ungateable_reason(agent)
    if reason is not None:
        raise ValueError(f"Recusal cannot gate agent {agent.name!r}: {reason}")
    return agent.clone(
        tools=gate_tools(list(agent.tools), policy, recording),
        handoffs=list(agent.handoffs),
        mcp_servers=list(agent.mcp_servers),
    )
