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
- a sub-agent behind ``agent.as_tool(...)`` whose own tools are not gated: the guardrail
  sees only the input handed to the sub-agent, never the calls it then makes. Gate the
  sub-agent first, ``gated_agent(sub, policy).as_tool(...)``, and the parent accepts it;
- the experimental Codex tool, which runs its own commands;
- ``Agent`` subclasses such as ``SandboxAgent``, which add tools of their own at run time.

What the guardrail could never judge, so ``gate_tools`` and ``gated_agent`` refuse the tool
when you call them, naming the tool and the cause (:func:`_tool_refusal`):

- a parameter the SDK prepares only by running the tool's own code (an ``Enum``, a model, a
  dataclass, a ``TypedDict``, a date, a tuple, a set, ``bytes``, a string pattern, a custom
  validator or a default factory): every call would be refused as unjudgeable;
- a variadic parameter (``*args``, ``**kwargs``), whose values have no names to judge by;
- a wrapper around the function that takes a parameter of its own, or takes one of the
  function's: it would run with a value no policy judged.

``gated_agent`` checks the agent when you call it. Build the agent completely, then gate
it; a tool appended to the gated agent's list afterwards is not gated. The gated copy has
its own ``tools``, ``handoffs`` and ``mcp_servers`` lists, so a later change to the
original agent's lists does not reach it.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
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
    parameter model only when validating it runs none of the tool's code."""
    model = schema.params_pydantic_model
    if can_prepare_without_user_code(model):
        return None
    try:
        fields = model.__pydantic_core_schema__["schema"]["fields"]
        names = [n for n, field in fields.items() if not _is_data_only_schema(field["schema"])]
    except Exception:  # noqa: BLE001, the refusal stands without the parameter's name
        names = []
    which = ("parameter " + ", ".join(f"`{n}`" for n in names)) if names else "a parameter"
    return (
        f"{which} has a type or a validation the SDK prepares only by running the tool's own "
        "code (an Enum, a model, a dataclass, a TypedDict, a date, a tuple, a set, bytes, a "
        "pattern, a custom validator or a default factory), so no call could be judged. "
        "Declare data-only parameters (str, int, float, bool, None, Literal, and lists and "
        "dicts of them) and resolve application types inside the tool body."
    )


def _wrapper_refusal(tool: FunctionTool, schema: FuncSchema) -> Optional[str]:
    """Why a callable on the way to the tool's code could run with a value no policy
    judged, or None.

    The SDK reads a ``functools.wraps`` wrapper as the function it wraps, prepares the
    arguments for that function's parameters and passes them to the wrapper, by position
    where it can. So every callable the call reaches (the kernel's ``reached_parameters``
    walk) must either take exactly the parameters the SDK prepared for, or forward
    ``*args`` and ``**kwargs`` and name none of its own. A wrapper that adds a parameter
    runs it at a default that is in no schema; one that takes a parameter of the function
    leaves the function at its own default; one that reorders them swaps the values."""
    get_callable = getattr(tool.on_invoke_tool, "_get_wrapped_callable", None)
    func = get_callable() if callable(get_callable) else None
    try:
        if not callable(func):
            raise TypeError("the invoker does not expose the function it wraps")
        levels = _reached_parameters(func)
    except Exception as exc:  # noqa: BLE001, a callable the gate cannot read is refused
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
    return _unprepared_parameters(schema) or _wrapper_refusal(tool, schema)


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
    here, before the policy runs, with or without a recording. So are arguments nested too
    deeply to read. Those refusals are bound to the argument text, as the one for
    malformed JSON is."""
    too_deep = "arguments are nested too deeply to judge"
    try:
        args = json.loads(arguments or "{}")
    except RecursionError:
        return None, _unjudgeable(tool_name, too_deep, arguments, record)
    except ValueError:
        return None, _unjudgeable(tool_name, "arguments are not valid JSON", arguments, record)
    try:
        args = _detached(args)
    except RecursionError:
        return None, _unjudgeable(tool_name, too_deep, arguments, record)
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
