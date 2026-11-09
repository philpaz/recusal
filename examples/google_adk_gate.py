"""
Offline Google ADK example: gate every tool call with a Recusal before-tool callback.

Requires Python 3.11+ (the ADK's minimum) and is tested with exactly google-adk==2.13.0 and
google-genai==2.34.0. Recusal itself remains Python 3.9+ and has no ADK dependency.

The hook is the ADK's own. An ``LlmAgent`` runs its ``before_tool_callback`` immediately
before each tool call, with the tool, the arguments the tool will receive and the tool
context; a callback that returns a dict answers the call instead of running the tool.
``recusal_before_tool(policy)`` judges the call with ``recusal.claude_code.decide`` and, on
a refusal, returns ``{"error": reason}`` so the model reads Recusal's reason on its next turn
and the tool never runs. Every call in a model turn is judged separately. The ADK routes
every ``BaseTool`` through the same path (function tools, long-running tools, agent tools,
MCP toolsets), so the callback sees each of them by name and arguments.

Two ways to attach it:

- ``gated_agent(agent, policy)`` returns a copy of the agent with the gate appended as the
  LAST before-tool callback, and gates its ``sub_agents`` and the agent behind every
  ``AgentTool`` the same way. Last, because callbacks run in order and a callback listed
  after the gate could change the arguments the gate judged; a callback listed before it
  that answers the call itself means the tool did not run.
- ``RecusalPlugin(policy)`` is the same gate as a runner plugin, for every agent the runner
  drives, so one plugin covers an app whose agents you do not build yourself. Plugin
  callbacks run before any agent's callbacks, so the plugin cannot be placed after an
  agent's own ``before_tool_callback``, which could change the arguments the plugin judged.
  Its contract is therefore explicit: it judges a call only when nothing it cannot see runs
  after it, that is, when the executing agent carries no before-tool callback other than a
  Recusal gate and no plugin listed after it overrides ``before_tool_callback``; otherwise it
  refuses the call, naming the callback or plugin, and the fix (``gated_agent`` for the
  agent, which appends the gate last among its callbacks; listing the plugin last). A call
  whose invocation context does not expose the executing agent is refused too.

What the callback cannot see, so ``gated_agent`` refuses rather than skips it:

- a ``code_executor`` on the agent: model-written code runs outside the tool path;
- a ``sub_agent`` or an ``AgentTool`` whose agent is not itself gated: its tool calls run
  under its own callbacks, not the parent's (``gated_agent`` gates them recursively, and
  refuses an agent class it does not know how to gate);
- an ``LlmAgent`` subclass: it can add behavior the gate was not written for.

No model, API key or network is used: ``ScriptedLlm`` proposes the calls from a script,
and run_sql only records SQL text.

Run after installing:
    pip install google-adk==2.13.0 google-genai==2.34.0
    python examples/google_adk_gate.py
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import os
import sys
import typing
from typing import Any, AsyncGenerator, Callable, List, Optional

GOOGLE_ADK_VERSION = "2.13.0"
GOOGLE_GENAI_VERSION = "2.34.0"

if sys.version_info < (3, 11):
    print(
        "google_adk_gate.py requires Python 3.11+ "
        f"(google-adk {GOOGLE_ADK_VERSION}); "
        f"this is Python {sys.version_info[0]}.{sys.version_info[1]}.",
        file=sys.stderr,
    )
    raise SystemExit(2)

try:
    from google.adk.agents import BaseAgent, LlmAgent
    from google.adk.agents.loop_agent import LoopAgent
    from google.adk.agents.parallel_agent import ParallelAgent
    from google.adk.agents.sequential_agent import SequentialAgent
    from google.adk.models import BaseLlm, LlmRequest, LlmResponse
    from google.adk.plugins import BasePlugin
    from google.adk.runners import InMemoryRunner
    from google.adk.tools import AgentTool, BaseTool, FunctionTool, ToolContext
    from google.genai import types
except ImportError as exc:
    raise SystemExit(
        "Install the example dependencies first: "
        f"google-adk=={GOOGLE_ADK_VERSION} google-genai=={GOOGLE_GENAI_VERSION}"
    ) from exc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.sql_scope_policy import policy as sql_scope_policy  # noqa: E402
from recusal.claude_code import decide  # noqa: E402

Policy = Callable[[str, dict], List[Any]]
EXECUTED: list[tuple[str, dict[str, Any]]] = []

#: Workflow agents run no tools of their own; their ``sub_agents`` do. ``gated_agent``
#: gates those and leaves the workflow agent's own class alone.
_WORKFLOW_AGENTS = (SequentialAgent, ParallelAgent, LoopAgent)


def echo(text: str) -> str:
    """Return harmless text."""
    EXECUTED.append(("echo", {"text": text}))
    return text


def run_sql(sql: str) -> str:
    """Record SQL text without connecting to a database."""
    EXECUTED.append(("run_sql", {"sql": sql}))
    return f"recorded SQL: {sql}"


def charge(amount: int = 100) -> str:
    """Record a hypothetical charge with a default the model may omit; contacts nothing."""
    EXECUTED.append(("charge", {"amount": amount}))
    return "recorded"


# --- the supported invocation: an explicit type set, defaults completed before judgment -------

_SCALARS = (str, int, float, bool)
_UNION_TYPE = type(int | None)  # the PEP 604 union, without shadowing google.genai.types


def _supported_annotation(annotation: Any) -> bool:
    """True for an annotation whose preparation is a pure function of the value: ``str``,
    ``int``, ``float``, ``bool``, ``Optional`` of one, ``list[T]`` and ``dict[str, T]`` with
    ``T`` supported in turn. Everything else (no annotation, ``Any``, a model, an ``Enum``, an
    ``Annotated``, a dataclass, a bare ``list``/``dict``) is unsupported."""
    if annotation in _SCALARS:
        return True
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is typing.Union or origin is _UNION_TYPE:
        rest = [a for a in args if a is not type(None)]
        return len(rest) == 1 and _supported_annotation(rest[0])
    if origin is list:
        return len(args) == 1 and _supported_annotation(args[0])
    if origin is dict:
        return len(args) == 2 and args[0] is str and _supported_annotation(args[1])
    return False


def _json_data(value: Any) -> bool:
    """True for a value in the JSON domain (what a default may be)."""
    if value is None or isinstance(value, _SCALARS):
        return True
    if isinstance(value, list):
        return all(_json_data(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_data(v) for k, v in value.items())
    return False


def _unsupported_signature(func: Callable[..., Any], skip: tuple[str, ...] = ()) -> Optional[str]:
    """Why this callable's parameters cannot all be judged, or None if they can."""
    try:
        parameters = inspect.signature(func).parameters
        # resolves string annotations (PEP 563) and keeps Annotated so it can be refused
        hints = typing.get_type_hints(func, include_extras=True)
    except Exception:  # noqa: BLE001, an uninspectable callable cannot be judged
        return "its signature cannot be inspected"
    for name, parameter in parameters.items():
        if name in skip:
            continue
        annotation = hints.get(name, parameter.empty)
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            return f"`{name}` is variadic; every argument the tool receives must be judged by name"
        if parameter.kind is parameter.POSITIONAL_ONLY:
            return f"`{name}` is positional-only and cannot be judged by name"
        if not _supported_annotation(annotation):
            return f"`{name}` is not annotated with a supported data type"
        if parameter.default is not parameter.empty and not _json_data(parameter.default):
            return f"`{name}` has a default outside the JSON domain"
    return None


#: The FunctionTool methods that prepare and run an invocation. A tool class is supported
#: only while every one of them is FunctionTool's own (the ADK's built-in subclasses such as
#: transfer_to_agent inherit them; a user subclass that overrides preprocessing is refused).
_STOCK_LIFECYCLE = (
    "_preprocess_args",
    "_preprocess_args_with_validation",
    "_validate_args",
    "run_async",
)


def _prepared(tool: Any, args: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Prepare the invocation the tool will execute, in place, and return a detached copy
    for judgment; or ``(None, reason)`` when it cannot be prepared faithfully.

    Supported: a ``google.adk.tools.FunctionTool`` whose class keeps FunctionTool's own
    preparation and run methods (``_STOCK_LIFECYCLE``; the ADK's built-in subclasses do, a
    user subclass that overrides preprocessing does not) and whose parameters, outside the
    ones the ADK injects (``_ignore_params``),
    all carry a supported data annotation and JSON defaults. Omitted parameters are completed
    from the signature's defaults INTO the raw arguments first, then the tool's own
    ``_preprocess_args_with_validation`` runs once, so a default such as ``int = "100"`` is
    prepared exactly as explicit input would be; the converted values are written back into
    ``args`` (the ADK calls the tool with this dict) and the policy judges a deep copy. On
    the supported set the ADK's execution-time preparation returns equal values, with the
    validation feature on (coercion) or off (none), so judged equals received. A validation
    failure is refused as such, without the SDK's text, which carries input values.
    """
    if not isinstance(tool, FunctionTool):
        return (
            None,
            f"only a google.adk.tools.FunctionTool can be judged, not {type(tool).__name__}",
        )
    for method in _STOCK_LIFECYCLE:
        if getattr(type(tool), method) is not getattr(FunctionTool, method):
            return (
                None,
                f"{type(tool).__name__} overrides FunctionTool.{method}, so its preparation "
                "is not the stock one",
            )
    func = tool.func
    skip = tuple(getattr(tool, "_ignore_params", ()) or ())
    why = _unsupported_signature(func, skip)
    if why is not None:
        return None, f"its signature cannot be judged ({why})"
    parameters = inspect.signature(func).parameters
    raw: dict[str, Any] = {}
    for name, parameter in parameters.items():
        if name in skip or name in args:
            continue
        if parameter.default is parameter.empty:
            return None, f"required parameter `{name}` was not supplied"
        raw[name] = copy.deepcopy(parameter.default)
    unknown = sorted(k for k in args if k not in parameters)
    if unknown:
        return None, f"unknown parameter(s) {unknown!r}"
    raw.update(args)
    converted, errors = tool._preprocess_args_with_validation(raw)
    if errors:
        return None, f"{len(errors)} argument(s) do not validate against the tool's parameters"
    args.clear()
    args.update(converted)
    return copy.deepcopy(converted), None


def _refusal(tool: Any, args: Any, policy: Policy) -> Optional[str]:
    """Return the reason a call is refused, or None to run it.

    The ADK hands the callback the same ``args`` dict the tool will be called with; the
    invocation is prepared in place (:func:`_prepared`) and the policy judges a detached
    copy of exactly what will run. A non-dict is refused: the tool would not accept it
    either, and the policy cannot read it.
    """
    tool_name = getattr(tool, "name", "?")
    if not isinstance(args, dict):
        return f"Recusal refused `{tool_name}`: arguments are not a mapping"
    try:
        invocation, why = _prepared(tool, args)
    except Exception as exc:  # noqa: BLE001, a preparation the gate cannot complete is refused
        return f"Recusal refused `{tool_name}`: the invocation cannot be prepared ({type(exc).__name__})"
    if invocation is None:
        return f"Recusal refused `{tool_name}`: {why}"
    try:
        decision, reason = decide(tool_name, invocation, policy, safe_errors=True)
    except Exception as exc:  # noqa: BLE001, a buggy policy must not silently disable the gate
        return f"Recusal failed closed (policy error): RECUSAL_POLICY_ERROR ({type(exc).__name__})"
    return reason if decision == "deny" else None


def recusal_before_tool(policy: Policy):
    """Return a ``before_tool_callback`` that judges each tool call before it runs.

    A refusal is returned as the tool's response, ``{"error": reason}``, so the model sees
    why on its next turn; ``None`` lets the ADK run the tool.
    """

    def gate(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext):
        reason = _refusal(tool, args, policy)
        if reason is not None:
            return {"error": reason}
        return None

    gate.recusal_gate = True  # type: ignore[attr-defined]
    return gate


class RecusalPlugin(BasePlugin):
    """The same gate as a runner plugin: every agent the runner drives.
    ``InMemoryRunner(agent, plugins=[..., RecusalPlugin(policy)])``.

    Plugin callbacks run before any agent's before-tool callbacks, and plugins run in list
    order, so the plugin judges a call only when nothing that could change the arguments
    runs after it (``_ordering_reason``); otherwise it refuses the call before judgment.
    """

    def __init__(self, policy: Policy, name: str = "recusal"):
        super().__init__(name=name)
        self._policy = policy

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> Optional[dict[str, Any]]:
        reason = _ordering_reason(self, tool_context)
        if reason is not None:
            return {"error": f"Recusal refused `{tool.name}`: {reason}"}
        reason = _refusal(tool, tool_args, self._policy)
        return {"error": reason} if reason is not None else None


def _ordering_reason(plugin: RecusalPlugin, tool_context: Any) -> Optional[str]:
    """Why the plugin is not the last before-tool seam for this call, or None if it is.

    The executing agent is read from the invocation context behind ``tool_context``; its
    before-tool callbacks run after every plugin, so any callback that is not a Recusal gate
    could change the arguments the plugin judged. A plugin listed after this one whose class
    overrides ``before_tool_callback`` runs after it too. Either is refused with the fix
    named; so is a context that does not expose the agent.
    """
    invocation = getattr(tool_context, "_invocation_context", None)
    agent = getattr(invocation, "agent", None)
    if agent is None:
        return "the invocation context does not expose the executing agent"
    if not isinstance(agent, LlmAgent):
        return (
            f"the executing agent {agent.name!r} is a {type(agent).__name__}, "
            "whose before-tool callbacks this plugin cannot read"
        )
    for callback in _callbacks(agent):
        if not getattr(callback, "recusal_gate", False):
            name = getattr(callback, "__name__", type(callback).__name__)
            return (
                f"agent {agent.name!r} runs a before_tool_callback ({name}) after this "
                "plugin; gate it with gated_agent(agent, policy)"
            )
    plugins = list(getattr(getattr(invocation, "plugin_manager", None), "plugins", []))
    position = next((i for i, p in enumerate(plugins) if p is plugin), None)
    for later in plugins[position + 1 :] if position is not None else []:
        if isinstance(later, RecusalPlugin):
            continue
        if type(later).before_tool_callback is not BasePlugin.before_tool_callback:
            return (
                f"plugin {later.name!r} runs a before_tool_callback after this plugin; "
                "list RecusalPlugin last"
            )
    return None


def _callbacks(agent: LlmAgent) -> list:
    existing = agent.before_tool_callback
    if existing is None:
        return []
    return list(existing) if isinstance(existing, list) else [existing]


def _carries_gate(agent: LlmAgent) -> bool:
    return any(getattr(cb, "recusal_gate", False) for cb in _callbacks(agent))


def _ungateable_reason(agent: Any) -> Optional[str]:
    """Why an agent can run something the callback never sees, or None if it cannot."""
    if isinstance(agent, _WORKFLOW_AGENTS):
        return None
    if type(agent) is not LlmAgent:
        return (
            f"{type(agent).__name__} is not a plain LlmAgent; a subclass can run tools or "
            "code the before-tool callback never sees"
        )
    if agent.code_executor is not None:
        return "it has a code_executor; model-written code runs outside the tool path"
    return None


def is_gated(agent: Any, _seen: Optional[set] = None) -> bool:
    """True if every tool call this agent, its sub-agents and its agent tools can make
    passes through a Recusal before-tool callback."""
    seen = set() if _seen is None else _seen
    if id(agent) in seen:
        return True
    seen.add(id(agent))
    if _ungateable_reason(agent) is not None:
        return False
    if isinstance(agent, LlmAgent):
        if not _carries_gate(agent):
            return False
        for tool in agent.tools:
            if isinstance(tool, AgentTool) and not is_gated(tool.agent, seen):
                return False
    return all(is_gated(sub, seen) for sub in agent.sub_agents)


def gated_agent(agent: BaseAgent, policy: Policy) -> BaseAgent:
    """Return a copy of the agent whose every tool call is judged by Recusal.

    The gate is appended as the LAST before-tool callback; ``sub_agents`` and the agent
    behind every ``AgentTool`` are gated the same way. Raises ValueError if the agent, or
    any agent under it, can run something the callback cannot see.
    """
    reason = _ungateable_reason(agent)
    if reason is not None:
        raise ValueError(f"Recusal cannot gate agent {agent.name!r}: {reason}")
    update: dict[str, Any] = {"sub_agents": [gated_agent(sub, policy) for sub in agent.sub_agents]}
    if isinstance(agent, LlmAgent):
        tools = []
        for tool in agent.tools:
            if isinstance(tool, AgentTool):
                tools.append(
                    AgentTool(
                        gated_agent(tool.agent, policy), skip_summarization=tool.skip_summarization
                    )
                )
            else:
                tools.append(tool)
        update["tools"] = tools
        update["before_tool_callback"] = _callbacks(agent) + [recusal_before_tool(policy)]
    return agent.clone(update=update)


# --- offline fixtures: a scripted model, no network --------------------------------------


class ScriptedLlm(BaseLlm):
    """A model that proposes the scripted calls on its first turn, then answers "done"."""

    calls: list[tuple[str, Any, str]] = []
    turns: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.turns += 1
        if self.turns == 1 and self.calls:
            parts = [
                types.Part(function_call=types.FunctionCall(id=call_id, name=name, args=args))
                for name, args, call_id in self.calls
            ]
        else:
            parts = [types.Part(text="done")]
        yield LlmResponse(content=types.Content(role="model", parts=parts))


def scripted_agent(calls: list[tuple[str, Any, str]], tools: Optional[list] = None) -> LlmAgent:
    return LlmAgent(
        name="assistant",
        model=ScriptedLlm(model="scripted", calls=list(calls)),
        tools=[FunctionTool(echo), FunctionTool(run_sql), FunctionTool(charge)]
        if tools is None
        else tools,
    )


async def run_agent(agent: BaseAgent, plugins: Optional[list] = None) -> dict[str, Any]:
    """Run the agent offline through InMemoryRunner; return each call's response by id."""
    runner = InMemoryRunner(agent=agent, app_name="recusal-adk-example", plugins=plugins)
    session = await runner.session_service.create_session(
        app_name="recusal-adk-example", user_id="user"
    )
    message = types.Content(role="user", parts=[types.Part(text="go")])
    responses: dict[str, Any] = {}
    async for event in runner.run_async(user_id="user", session_id=session.id, new_message=message):
        for part in event.content.parts if event.content and event.content.parts else []:
            if part.function_response is not None:
                responses[part.function_response.id] = part.function_response.response
    return responses


def run_calls(
    calls: list[tuple[str, Any, str]],
    *,
    policy: Policy = sql_scope_policy,
    gated: bool = True,
    via_plugin: bool = False,
    tools: Optional[list] = None,
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    """Propose the calls to an agent with echo, run_sql and charge; return responses and
    what ran."""
    EXECUTED.clear()
    agent: BaseAgent = scripted_agent(calls, tools=tools)
    plugins = None
    if gated and via_plugin:
        plugins = [RecusalPlugin(policy)]
    elif gated:
        agent = gated_agent(agent, policy)
    responses = asyncio.run(run_agent(agent, plugins=plugins))
    return responses, list(EXECUTED)


def main() -> None:
    calls = [
        ("echo", {"text": "hello"}, "c1"),
        ("run_sql", {"sql": "DELETE FROM audit_log"}, "c2"),
        ("run_sql", {"sql": "SELECT * FROM audit_log"}, "c3"),
    ]
    for label, via_plugin in (("before_tool_callback", False), ("RecusalPlugin", True)):
        responses, executed = run_calls(calls, via_plugin=via_plugin)
        print(f"{label}:")
        for name, _, call_id in calls:
            print(f"  {name} ({call_id}): {responses[call_id]}")
        print(f"  tools that ran: {len(executed)}")


if __name__ == "__main__":
    main()
