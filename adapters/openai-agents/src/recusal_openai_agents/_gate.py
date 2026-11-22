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
  sub-agent first, ``gated_agent(sub, policy).as_tool(...)``, and the parent accepts it;
- the experimental Codex tool, which runs its own commands;
- ``Agent`` subclasses such as ``SandboxAgent``, which add tools of their own at run time.

What the guardrail could never judge, so ``gate_tools`` and ``gated_agent`` refuse the tool
when you call them, naming the tool and the cause (:func:`_tool_refusal`):

- a parameter the SDK prepares only by running the tool's own code (an ``Enum``, a model, a
  dataclass, a ``TypedDict``, a date, a tuple, a set, ``bytes``, a string pattern, a custom
  validator, a default factory other than ``list`` or ``dict``, a default that is not JSON
  data): every call would be refused as unjudgeable;
- a mapping parameter whose key type is not ``str``: the SDK converts the keys, and
  arguments are judged as JSON data, whose keys are strings;
- a variadic parameter (``*args``, ``**kwargs``), whose values have no names to judge by;
- a wrapper around the function that takes a parameter of its own, or takes one of the
  function's: it would run with a value no policy judged;
- a ``functools.singledispatch`` function, which picks the code that runs by the type of
  an argument, and a decorator whose signature cannot be read.

Which callables are read for that: the function, what its ``__wrapped__`` chain leads to,
a bound method's function and a callable object's ``__call__``. A wrapper that calls a
function it holds some other way (in a closure, in an attribute) is not read.

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
from collections.abc import Mapping
from typing import Any, Callable, List, Optional

from agents import (
    Agent,
    FunctionTool,
    ToolGuardrailFunctionOutput,
    tool_input_guardrail,
)
from agents._function_tool_arguments import _is_data_only_schema, can_prepare_without_user_code
from agents.function_schema import FuncSchema

from recusal import Finding, decide
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
        "code (an Enum, a model, a dataclass, a TypedDict, a date, a tuple, a set, bytes, a "
        "pattern, a custom validator), so no call could be judged. "
        "Declare data-only parameters (str, int, float, bool, None, Literal, and lists and "
        "dicts of them) and resolve application types inside the tool body."
    )


def _quoted(names: list[str]) -> str:
    return ", ".join(f"`{name}`" for name in names)


def _other_keys(node: Any) -> bool:
    """Whether a parameter's core schema holds a mapping whose key type is not ``str``."""
    if type(node) is dict:
        keys = node.get("keys_schema")
        if node.get("type") == "dict" and type(keys) is dict:
            if keys.get("type") not in ("str", "any"):
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
        f"parameter {_quoted(names)} is a mapping whose key type is not `str`. The SDK "
        "would convert the keys of the JSON object the model sends, and the gate judges "
        "arguments as JSON data, whose keys are strings, so no call that gives it a member "
        "could be judged. Declare `dict[str, ...]` and convert the keys inside the tool body."
    )


#: The object ``functools.lru_cache`` returns, which the kernel's walk reads through.
_LRU_WRAPPER = type(functools.lru_cache(maxsize=None)(lambda: None))

_SINGLEDISPATCH = (
    "it is a `functools.singledispatch` function: the implementation that runs is chosen "
    "by the type of an argument and each one has its own parameters, so the gate cannot "
    "tell which would run or with what defaults. Register one plain function per tool and "
    "dispatch inside its body."
)

_OPAQUE_DECORATOR = (
    "a decorator on the function cannot be inspected (a class-based decorator that is also "
    "a descriptor, such as `cachetools.cached`), so the functions under it cannot be read. "
    "Call the decorated function from a plain function and register that."
)


def _wrapped_chain(func: Any) -> list[Any]:
    """``func`` and the callables its ``__wrapped__`` attributes lead to, outermost first;
    a ``functools.partial`` is followed to its function."""
    chain: list[Any] = []
    while func is not None and len(chain) < 64 and not any(func is link for link in chain):
        chain.append(func)
        if isinstance(func, functools.partial):
            func = func.func
        else:
            func = getattr(func, "__wrapped__", None)
    return chain


def _dispatches_by_type(link: Any) -> bool:
    """A ``functools.singledispatch`` function, or a ``singledispatchmethod`` read from an
    object: what it runs depends on the type of an argument."""
    if callable(getattr(link, "dispatch", None)) and hasattr(link, "registry"):
        return True
    register = getattr(link, "register", None)
    return isinstance(getattr(register, "__self__", None), functools.singledispatchmethod)


def _opaque_decorator(link: Any) -> bool:
    """A decorator object ``inspect`` reads as a built-in and finds no signature for: an
    instance of a class with ``__get__``, as ``cachetools.cached`` returns."""
    if not inspect.ismethoddescriptor(link) or isinstance(link, _LRU_WRAPPER):
        return False
    try:
        inspect.signature(link, follow_wrapped=False)
    except ValueError:
        return True
    except TypeError:
        return False
    return False


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

    The walk reads the callable, what ``__wrapped__`` leads to, a bound method's function
    and a callable object's ``__call__``. A ``functools.singledispatch`` function is
    refused: which implementation runs depends on the type of an argument, and the walk
    reads only the one ``__wrapped__`` names. A wrapper that calls a function it holds in
    a closure or an attribute is not read, and is not refused."""
    get_callable = getattr(tool.on_invoke_tool, "_get_wrapped_callable", None)
    func = get_callable() if callable(get_callable) else None
    chain = _wrapped_chain(func)
    if any(_dispatches_by_type(link) for link in chain):
        return _SINGLEDISPATCH
    try:
        if not callable(func):
            raise TypeError("the invoker does not expose the function it wraps")
        levels = _reached_parameters(func)
    except Exception as exc:  # noqa: BLE001, a callable the gate cannot read is refused
        if type(exc) is ValueError and any(_opaque_decorator(link) for link in chain):
            return _OPAQUE_DECORATOR
        return f"the function it runs cannot be inspected ({type(exc).__name__})"
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
    ``(None, reason)`` for text that is not one.

    A non-finite number (``NaN``, ``Infinity``, a literal too large for a float) is not
    JSON data: no policy can compare it and no record can fingerprint it, so it is refused
    here, before the policy runs, with or without a recording. So are arguments that hold
    a value more than ``_MAX_DEPTH`` levels down, before anything else reads them. Those
    refusals are bound to the argument text, as the one for malformed JSON is."""
    try:
        args = json.loads(arguments or "{}")
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
        # agent.as_tool(): no parameter model, the SDK hands the input text to the sub-agent
        # as is, and the sub-agent's own gate (which gate_tools requires) judges its calls
        try:
            decision, reason = decide(tool_name, args, policy, safe_errors=True, **record)
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

    The SDK merges ``extra_body`` into the body of every model request and passes
    ``extra_args`` as request arguments, so a ``tools`` key there (or a stored ``prompt``)
    reaches the provider without being in the agent's ``tools`` list. Only the agent's own
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
                f"model_settings.{field} sets {_quoted(found)}, which puts tools into the "
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
    running the tool's code, a variadic parameter, a wrapper with a parameter of its own).
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
        if getattr(tool, "_is_agent_tool", False) and not agent_gated(
            getattr(tool, "_agent_instance", None)
        ):
            raise TypeError(
                f"Recusal cannot gate agent tool {name!r}: its agent's own tool calls are not "
                "gated. Use gated_agent(sub_agent, policy).as_tool(...)"
            )
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
