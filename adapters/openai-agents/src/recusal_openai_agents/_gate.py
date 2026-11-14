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
- a sub-agent behind ``agent.as_tool(...)`` whose own tools are not gated: the guardrail
  sees only the input handed to the sub-agent, never the calls it then makes. Gate the
  sub-agent first, ``gated_agent(sub, policy).as_tool(...)``, and the parent accepts it;
- the experimental Codex tool, which runs its own commands;
- ``Agent`` subclasses such as ``SandboxAgent``, which add tools of their own at run time.

``gated_agent`` checks the agent when you call it. Build the agent completely, then gate
it; a tool appended to the gated agent's list afterwards is not gated.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import typing
from typing import Any, Callable, List, Optional

from agents import (
    Agent,
    FunctionTool,
    RunContextWrapper,
    ToolGuardrailFunctionOutput,
    tool_input_guardrail,
)
from agents.tool_context import ToolContext
from pydantic_core import to_jsonable_python

from recusal import Finding, decide

#: A Recusal policy: ``policy(tool_name, tool_input)`` returns a list of findings.
Policy = Callable[[str, dict], List[Any]]
#: The name the guardrail carries, which is how a gated tool is recognized.
GUARDRAIL_NAME = "recusal"


def _context_param(func: Any) -> Optional[str]:
    """The name of ``func``'s leading context parameter (``RunContextWrapper`` or
    ``ToolContext``), which the SDK fills itself, or None when there is none."""
    params = list(inspect.signature(func).parameters.values())
    if not params:
        return None
    first = params[0]
    try:
        hint = typing.get_type_hints(func).get(first.name, first.annotation)
    except Exception:  # noqa: BLE001, unresolvable hints fall back to the raw annotation
        hint = first.annotation
    origin = typing.get_origin(hint) or hint
    if inspect.isclass(origin) and issubclass(origin, (RunContextWrapper, ToolContext)):
        return first.name
    if isinstance(hint, str) and ("RunContextWrapper" in hint or "ToolContext" in hint):
        return first.name
    return None


def _effective_arguments(tool: FunctionTool, prepared: Any) -> dict[str, Any]:
    """Name the invocation the SDK prepared: positional values bound to the wrapped
    function's parameters (the context parameter excluded), keyword values as they are,
    every value projected to JSON data (a nested model becomes its dict), so the policy and
    the audit fingerprint read exactly what the function will receive."""
    get_callable = getattr(tool.on_invoke_tool, "_get_wrapped_callable", None)
    func = get_callable() if callable(get_callable) else None
    if not callable(func):
        raise TypeError("the invoker does not expose the function it wraps")
    ctx_name = _context_param(func)
    names: list[str] = []
    for param in inspect.signature(func).parameters.values():
        if param.name == ctx_name:
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError(f"parameter *{param.name} cannot be judged by name")
        names.append(param.name)
    if len(prepared.args) > len(names):
        raise TypeError("more positional values than named parameters")
    effective = dict(zip(names, prepared.args))
    effective.update(prepared.kwargs)
    return {name: to_jsonable_python(value) for name, value in effective.items()}


def _uninspectable(prepared: Any) -> bool:
    """The SDK prepares nothing (empty, and not "unchanged") when the parameter model runs
    user code (a custom validator, a default factory) or the arguments do not validate."""
    return not prepared.args and not prepared.kwargs and not prepared.unchanged


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

    This is the right seam only where nothing validates or transforms the arguments
    between here and execution (the hosted MCP approval callback: OpenAI runs the call
    with exactly these arguments). Function tools go through
    :func:`_function_tool_refusal`, which judges the invocation the SDK prepared.
    ``record`` goes straight to ``decide``, which then writes the decision to an audit log.
    """
    try:
        args = json.loads(arguments or "{}")
    except ValueError:
        return _unjudgeable(tool_name, "arguments are not valid JSON", arguments, record)
    if not isinstance(args, dict):
        return _unjudgeable(tool_name, "arguments are not a JSON object", args, record)
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
    because preparing it would run the tool's code before the gate; such a call is refused,
    since the value it would execute cannot be judged. Validate inside the tool body and
    declare data-only parameters (scalars, lists and dicts of them).

    ``record`` goes straight to ``decide``, so a gate that puts the decision on a log
    (``audited_agent``, ``audited_tools``) decides, and refuses, exactly as this one does. The
    entry for a judged call carries the fingerprint of the prepared invocation, named and
    projected to JSON data; a call refused before judgment is bound to the arguments as
    the model proposed them.
    """
    try:
        args = json.loads(arguments or "{}")
    except ValueError:
        return _unjudgeable(tool_name, "arguments are not valid JSON", arguments, record)
    if not isinstance(args, dict):
        return _unjudgeable(tool_name, "arguments are not a JSON object", args, record)
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
    prepare = getattr(tool.on_invoke_tool, "prepare_arguments", None)
    if not callable(prepare):
        return _unjudgeable(
            tool_name,
            "its invoker does not expose the arguments it would execute; build the tool with "
            "function_tool",
            args,
            record,
        )
    prepared = prepare(arguments, tool_name)
    if prepared is None or _uninspectable(prepared):
        return _unjudgeable(
            tool_name,
            "the arguments the tool would execute cannot be judged (its schema runs custom "
            "validation, or the arguments do not validate); validate inside the tool body or "
            "declare a data-only schema",
            args,
            record,
        )
    try:
        effective = _effective_arguments(tool, prepared)
    except Exception as exc:  # noqa: BLE001, an invocation the gate cannot name is refused
        return _unjudgeable(
            tool_name, f"cannot name its validated arguments ({type(exc).__name__})", args, record
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
    duplicates = _duplicate_names(agent.tools)
    if duplicates:
        return (
            f"its function tools share a name {duplicates!r}; the SDK resolves a shared name "
            "last-wins at execution, so a guardrail could judge one tool while another runs"
        )
    return None


def is_gated(agent: Any, _seen: set[int] | None = None) -> bool:
    """True if every tool this agent can run passes through a Recusal guardrail."""
    seen = set() if _seen is None else _seen
    if id(agent) in seen:
        return True
    seen.add(id(agent))
    if _ungateable_reason(agent) is not None:
        return False
    for tool in agent.tools:
        if not isinstance(tool, FunctionTool) or not _carries_gate(tool):
            return False
        if getattr(tool, "_is_codex_tool", False):
            return False
        if getattr(tool, "_is_agent_tool", False):
            if not is_gated(getattr(tool, "_agent_instance", None), seen):
                return False
    return True


def gate_tools(
    tools: list[Any], policy: Policy, recording: Optional[Recording] = None
) -> list[FunctionTool]:
    """Return copies of the tools with the Recusal guardrail added; the originals are unchanged.

    Raises TypeError, naming the tool, for anything the guardrail cannot see. ``recording``
    puts every decision of every guardrail it attaches on an audit log.
    """
    duplicates = _duplicate_names(tools)
    if duplicates:
        raise TypeError(
            f"Recusal cannot gate tools that share a name {duplicates!r}: the SDK resolves a "
            "shared name last-wins at execution, so a guardrail could judge one tool while "
            "another runs; give every function tool a unique name"
        )
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
        if not getattr(tool, "_is_agent_tool", False) and not callable(
            getattr(tool.on_invoke_tool, "prepare_arguments", None)
        ):
            raise TypeError(
                f"Recusal cannot gate {name!r}: its invoker does not expose the arguments it "
                "would execute, so the gate could judge only the model's raw JSON. Build the "
                "tool with function_tool."
            )
        if getattr(tool, "_is_agent_tool", False) and not is_gated(
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

    Raises ValueError if the agent can run anything the guardrail cannot see.
    """
    reason = _ungateable_reason(agent)
    if reason is not None:
        raise ValueError(f"Recusal cannot gate agent {agent.name!r}: {reason}")
    return agent.clone(tools=gate_tools(list(agent.tools), policy, recording))
