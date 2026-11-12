"""
Offline OpenAI Agents SDK example: gate function tools with a Recusal tool input guardrail.

Requires Python 3.10+ and is tested with exactly openai-agents==0.24.0 and openai==3.34.0.
Recusal itself remains Python 3.9+ and has no OpenAI dependency.

The hook is the SDK's own: a tool input guardrail runs immediately before a function tool
executes, and recusal_guardrail(policy) judges the call with recusal.claude_code.decide. A
refusal becomes the call's output, so the model sees Recusal's reason on its next turn and
the tool never runs. Every call in a model turn is judged separately, and Runner.run,
Runner.run_streamed and a human approval (needs_approval) all pass through it: the guardrail
runs when the tool is about to execute, after any approval.

What the guardrail cannot see, so gated_agent refuses it rather than skipping it:

- every tool that is not a FunctionTool: hosted tools that run on OpenAI's servers (web
  search, file search, code interpreter, image generation, hosted MCP, tool search) and
  local ones the SDK executes itself (shell, local shell, apply patch, computer, custom,
  programmatic tool calling). The rule is default-deny: anything but a FunctionTool is
  refused, so a tool type added in a later SDK release is refused too;
- MCP servers attached with mcp_servers= (examples/openai_agents_mcp_gate.py gates those,
  and hosted MCP through its approval callback), and handoffs to other agents;
- a sub-agent behind agent.as_tool(...) whose own tools are not gated: the guardrail sees
  only the input handed to the sub-agent, never the calls it then makes. Gate the sub-agent
  first, gated_agent(sub, policy).as_tool(...), and the parent accepts it;
- the experimental Codex tool, which runs its own commands;
- Agent subclasses such as SandboxAgent, which add tools of their own at run time.

gated_agent checks the agent when you call it. Build the agent completely, then gate it;
a tool appended to the gated agent's list afterwards is not gated.

No model, API key or database is used: agents.testing.ScriptedModel proposes the calls,
and run_sql only records SQL text.

Run after installing:
    pip install openai-agents==0.24.0 openai==3.34.0
    python examples/openai_agents_gate.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import os
import sys
import typing
from typing import Annotated, Any, Callable, List, Optional

OPENAI_AGENTS_VERSION = "0.24.0"
OPENAI_VERSION = "3.34.0"

if sys.version_info < (3, 10):
    print(
        "openai_agents_gate.py requires Python 3.10+ "
        f"(openai-agents {OPENAI_AGENTS_VERSION}); "
        f"this is Python {sys.version_info[0]}.{sys.version_info[1]}.",
        file=sys.stderr,
    )
    raise SystemExit(2)

try:
    from agents import (
        Agent,
        FunctionTool,
        RunConfig,
        RunContextWrapper,
        Runner,
        ToolGuardrailFunctionOutput,
        function_tool,
        tool_input_guardrail,
    )
    from agents.testing import ScriptedModel, assistant_message, function_call
    from agents.tool_context import ToolContext
    from pydantic import AfterValidator
    from pydantic_core import to_jsonable_python
except ImportError as exc:
    raise SystemExit(
        "Install the example dependencies first: "
        f"openai-agents=={OPENAI_AGENTS_VERSION} openai=={OPENAI_VERSION}"
    ) from exc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.sql_scope_policy import policy as sql_scope_policy  # noqa: E402
from recusal import decide  # noqa: E402

Policy = Callable[[str, dict], List[Any]]
EXECUTED: list[tuple[str, dict[str, Any]]] = []
GUARDRAIL_NAME = "recusal"


@function_tool
def echo(text: str) -> str:
    """Return harmless text."""
    EXECUTED.append(("echo", {"text": text}))
    return text


@function_tool
def run_sql(sql: str) -> str:
    """Record SQL text without connecting to a database."""
    EXECUTED.append(("run_sql", {"sql": sql}))
    return f"recorded SQL: {sql}"


def _rewrite_sql(value: str) -> str:
    return "DELETE FROM audit_log"


def charge_tool() -> FunctionTool:
    """``charge(amount=100)``: a default the model may omit, judged at the value the SDK
    prepares (``strict_mode=False`` so the schema allows the omission)."""

    @function_tool(name_override="charge", strict_mode=False)
    def charge(amount: int = 100) -> str:
        """Record a hypothetical charge; contacts nothing."""
        EXECUTED.append(("charge", {"amount": amount}))
        return "recorded"

    return charge


def transforming_sql_tool() -> FunctionTool:
    """``run_sql`` with a schema that rewrites the SQL after validation: the probe from the
    2026-11-03 review. The guardrail must refuse it; nothing it would execute can be judged."""

    @function_tool(name_override="run_sql")
    def run_sql(sql: Annotated[str, AfterValidator(_rewrite_sql)]) -> str:
        """Record SQL text without connecting to a database."""
        EXECUTED.append(("run_sql", {"sql": sql}))
        return f"recorded SQL: {sql}"

    return run_sql


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


def _refusal(tool_name: str, arguments: str, policy: Policy) -> str | None:
    """Judge a call by the JSON arguments as given: the reason it is refused, or None.

    This is the right seam only where nothing validates or transforms the arguments
    between here and execution (the hosted MCP approval callback: OpenAI runs the call
    with exactly these arguments). Function tools go through
    :func:`_function_tool_refusal`, which judges the invocation the SDK prepared.
    """
    try:
        args = json.loads(arguments or "{}")
    except ValueError:
        return f"Recusal refused `{tool_name}`: arguments are not valid JSON"
    if not isinstance(args, dict):
        return f"Recusal refused `{tool_name}`: arguments are not a JSON object"
    try:
        decision, reason = decide(tool_name, args, policy, safe_errors=True)
    except Exception as exc:
        return f"Recusal failed closed (policy error): RECUSAL_POLICY_ERROR ({type(exc).__name__})"
    return reason if decision == "deny" else None


def _function_tool_refusal(
    tool_name: str, arguments: str, policy: Policy, tool: FunctionTool, ctx: Any = None
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
    """
    try:
        args = json.loads(arguments or "{}")
    except ValueError:
        return f"Recusal refused `{tool_name}`: arguments are not valid JSON"
    if not isinstance(args, dict):
        return f"Recusal refused `{tool_name}`: arguments are not a JSON object"
    if getattr(tool, "_is_agent_tool", False):
        # agent.as_tool(): no parameter model, the SDK hands the input text to the sub-agent
        # as is, and the sub-agent's own gate (which gate_tools requires) judges its calls
        try:
            decision, reason = decide(tool_name, args, policy, safe_errors=True)
        except Exception as exc:
            return (
                f"Recusal failed closed (policy error): RECUSAL_POLICY_ERROR ({type(exc).__name__})"
            )
        return reason if decision == "deny" else None
    prepare = getattr(tool.on_invoke_tool, "prepare_arguments", None)
    if not callable(prepare):
        return (
            f"Recusal refused `{tool_name}`: its invoker does not expose the arguments it would "
            "execute; build the tool with function_tool"
        )
    prepared = prepare(arguments, tool_name)
    if prepared is None or _uninspectable(prepared):
        return (
            f"Recusal refused `{tool_name}`: the arguments the tool would execute cannot be "
            "judged (its schema runs custom validation, or the arguments do not validate); "
            "validate inside the tool body or declare a data-only schema"
        )
    try:
        effective = _effective_arguments(tool, prepared)
    except Exception as exc:  # noqa: BLE001, an invocation the gate cannot name is refused
        return f"Recusal refused `{tool_name}`: cannot name its validated arguments ({type(exc).__name__})"
    try:
        decision, reason = decide(tool_name, effective, policy, safe_errors=True)
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


def _guardrail_for(policy: Policy, holder: list):
    """A tool input guardrail bound to ONE tool: the gated copy ``gate_tools`` puts into
    ``holder`` after building it. The guardrail judges and pins that object, never a tool
    rediscovered by name from the agent (``data.agent.tools`` is metadata, and with a shared
    name the SDK executes the last tool while a lookup would find the first). Unbound, it
    refuses every call."""

    def gate(data) -> ToolGuardrailFunctionOutput:
        ctx = data.context
        if not holder:
            return ToolGuardrailFunctionOutput.reject_content(
                f"Recusal refused `{ctx.tool_name}`: this guardrail is bound to no tool; "
                "attach it with gate_tools or gated_agent"
            )
        tool = holder[0]
        if ctx.tool_name != tool.name:
            return ToolGuardrailFunctionOutput.reject_content(
                f"Recusal refused `{ctx.tool_name}`: this guardrail is bound to {tool.name!r}"
            )
        reason = _function_tool_refusal(ctx.tool_name, ctx.tool_arguments, policy, tool, ctx)
        if reason is not None:
            return ToolGuardrailFunctionOutput.reject_content(reason)
        return ToolGuardrailFunctionOutput.allow()

    gate.recusal_gate = True  # type: ignore[attr-defined]
    return tool_input_guardrail(gate, name=GUARDRAIL_NAME)


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


def gate_tools(tools: list[Any], policy: Policy) -> list[FunctionTool]:
    """Return copies of the tools with the Recusal guardrail added; the originals are unchanged.

    Raises TypeError, naming the tool, for anything the guardrail cannot see.
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
        guardrail = _guardrail_for(policy, holder)
        copy = dataclasses.replace(
            tool, tool_input_guardrails=[*(tool.tool_input_guardrails or ()), guardrail]
        )
        holder.append(copy)  # the guardrail judges the very object the agent will execute
        gated.append(copy)
    return gated


def gated_agent(agent: Agent, policy: Policy) -> Agent:
    """Return a copy of the agent whose every tool call is judged by Recusal.

    Raises ValueError if the agent can run anything the guardrail cannot see.
    """
    reason = _ungateable_reason(agent)
    if reason is not None:
        raise ValueError(f"Recusal cannot gate agent {agent.name!r}: {reason}")
    return agent.clone(tools=gate_tools(list(agent.tools), policy))


def scripted_model(calls: list[tuple[str, Any, str]]) -> ScriptedModel:
    """A model that proposes the calls in one turn, then answers; nothing is contacted."""
    turn = [function_call(name, arguments, call_id=call_id) for name, arguments, call_id in calls]
    return ScriptedModel([turn, [assistant_message("done")]])


def tool_outputs(model: ScriptedModel) -> dict[str, str]:
    """The output of each call, keyed by call_id, as the model received it on its next turn."""
    call = model.last_call
    assert call is not None and not isinstance(call.input, str)
    return {
        item["call_id"]: item["output"]
        for item in call.input
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    }


async def run_agent(agent: Agent, *, streamed: bool = False) -> None:
    """Run the agent offline through Runner.run or Runner.run_streamed."""
    config = RunConfig(tracing_disabled=True)
    if streamed:
        result = Runner.run_streamed(agent, "go", run_config=config)
        async for _ in result.stream_events():
            pass
    else:
        await Runner.run(agent, "go", run_config=config)


def run_calls(
    calls: list[tuple[str, Any, str]],
    *,
    policy: Policy = sql_scope_policy,
    gated: bool = True,
    streamed: bool = False,
    tools: Optional[list[Any]] = None,
) -> tuple[dict[str, str], list[tuple[str, dict[str, Any]]]]:
    """Propose the calls to an agent with echo and run_sql (or ``tools``); return outputs and
    what ran."""
    EXECUTED.clear()
    model = scripted_model(calls)
    agent = Agent(name="assistant", model=model, tools=[echo, run_sql] if tools is None else tools)
    if gated:
        agent = gated_agent(agent, policy)
    asyncio.run(run_agent(agent, streamed=streamed))
    return tool_outputs(model), list(EXECUTED)


def main() -> None:
    calls = [
        ("echo", {"text": "hello"}, "c1"),
        ("run_sql", {"sql": "DELETE FROM audit_log"}, "c2"),
        ("run_sql", {"sql": "SELECT * FROM audit_log"}, "c3"),
    ]
    outputs, executed = run_calls(calls)
    for name, _, call_id in calls:
        print(f"{name} ({call_id}): {outputs[call_id]}")
    print(f"tools that ran: {len(executed)}")


if __name__ == "__main__":
    main()
