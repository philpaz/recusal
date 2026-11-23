"""The gate: one refusal function, and the two ways to attach it to a Google ADK agent.

The hook is the ADK's own. An ``LlmAgent`` runs its ``before_tool_callback`` immediately
before each tool call, with the tool, the arguments the tool will receive and the tool
context; a callback that returns a dict answers the call instead of running the tool.
``recusal_before_tool(policy)`` judges the call with ``recusal.decide`` and, on a refusal,
returns ``{"error": reason}`` so the model reads Recusal's reason on its next turn and the
tool never runs. Every call in a model turn is judged separately.

Two ways to attach it:

- ``gated_agent(agent, policy)`` returns a copy of the agent with the gate appended as the
  LAST before-tool callback, and gates its ``sub_agents`` and the agent behind every
  ``AgentTool`` the same way. Last, because callbacks run in order and a callback listed
  after the gate could change the arguments the gate judged; a callback listed before it
  that answers the call itself means the tool did not run.
- ``RecusalPlugin(policy)`` is the same gate as a runner plugin, for every agent the runner
  drives. Plugin callbacks run before any agent's callbacks, so the plugin cannot be placed
  after an agent's own ``before_tool_callback``, which could change the arguments the
  plugin judged. Its contract is therefore explicit: it judges a call only when nothing it
  cannot see runs after it, that is, when the executing agent carries no before-tool
  callback other than a Recusal gate and no plugin listed after it overrides
  ``before_tool_callback``; otherwise it refuses the call, naming the callback or plugin,
  and the fix. A call whose invocation context does not expose the executing agent is
  refused too.

What the callback cannot see, so ``gated_agent`` refuses rather than skips it:

- a ``code_executor`` on the agent: model-written code runs outside the tool path;
- a tool that is not a function, a ``FunctionTool`` or a plain ``AgentTool``: the built-in
  tools the model provider runs itself (``google_search``, ``url_context``, ...) never
  reach a callback, and toolsets (MCP, OpenAPI) cannot be judged; tools declared in
  ``generate_content_config`` run on the provider's side too;
- a ``sub_agent`` or an ``AgentTool`` whose agent is not itself gated: its tool calls run
  under its own callbacks, not the parent's (``gated_agent`` gates them recursively, and
  refuses an agent class it does not know how to gate);
- an ``LlmAgent`` subclass: it can add behavior the gate was not written for.
"""

from __future__ import annotations

import copy
import inspect
import re
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.llm.task._finish_task_tool import FinishTaskTool
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.plugins import BasePlugin
from google.adk.tools import AgentTool, BaseTool, FunctionTool, ToolContext
from google.adk.tools.agent_tool import _get_input_schema as _agent_input_schema
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.load_artifacts_tool import LoadArtifactsTool
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.adk.tools.set_model_response_tool import SetModelResponseTool

import recusal.invocation as _invocation
from recusal import Finding, decide
from recusal.invocation import detached as _detached
from recusal.invocation import json_data as _json_data
from recusal.invocation import reached_parameters as _reached_parameters
from recusal.invocation import unsupported_signature as _unsupported_signature

if TYPE_CHECKING:
    from google.adk.agents.llm_agent import BeforeToolCallback

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
#: than 0.15.0).
_HiddenWrapper = _walk_refusal(_invocation, "HiddenWrapper")
#: A single-dispatch function on the way can run more than its base (the same kernels).
_DispatchedAtCallTime = _walk_refusal(_invocation, "DispatchedAtCallTime")
#: Given the call's ``ToolContext``, the recording arguments ``decide`` takes for it
#: (``audit=``, ``surface=``, ``tool_use_id=``, ...). ``recusal_google_adk.recording``
#: builds one.
Recording = Callable[[Any], dict]

#: Workflow agents run no tools of their own; their ``sub_agents`` do. ``gated_agent``
#: gates those and leaves the workflow agent's own class alone.
_WORKFLOW_AGENTS = (SequentialAgent, ParallelAgent, LoopAgent)


#: The methods ``FunctionTool`` defines that do not take part in running a call: how the tool
#: is declared to the model, and a telemetry hook. ADK's own subclasses override only these
#: (``LongRunningFunctionTool``, ``TransferToAgentTool``, ``LoadMemoryTool``), besides
#: ``run_async`` (``AuthenticatedFunctionTool``, ``GoogleTool``, ``ComputerUseTool``), which
#: changes how the call runs and is refused.
_NOT_ON_THE_RUN_PATH = frozenset({"__init__", "_get_declaration", "_detect_error_in_response"})

#: Every other method ``FunctionTool`` defines prepares, filters, checks or runs the call
#: (``_preprocess_args_with_validation``, ``_prepare_invocation_args``,
#: ``check_require_confirmation``, ``_invoke_callable``, ``_run_generator_as_node``, ...). A
#: tool is supported only while each is ``FunctionTool``'s own, on its class and not set on
#: the object: an override of any of them can change the arguments after the policy judged
#: them. Read from ``FunctionTool`` itself, so a method added in a later ADK release is held
#: to the rule too.
_STOCK_LIFECYCLE = tuple(
    sorted(
        name
        for name, value in vars(FunctionTool).items()
        if callable(value) and name not in _NOT_ON_THE_RUN_PATH
    )
)


def _prepared(tool: Any, args: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Prepare the invocation the tool will execute, in place, and return a detached copy
    for judgment; or ``(None, reason)`` when it cannot be prepared faithfully.

    Supported: a ``google.adk.tools.FunctionTool`` whose class keeps FunctionTool's own
    preparation and run methods (``_STOCK_LIFECYCLE``; the ADK's built-in subclasses do, a
    user subclass that overrides preprocessing does not) and whose parameters, outside the
    ones the ADK injects (``_ignore_params``),
    all carry a supported data annotation and JSON defaults. Omitted parameters are completed
    from the signature's defaults INTO the raw arguments, then validated against the
    function's types (``_always_validated``, whatever the ADK's own setting), so a default
    such as ``int = "100"`` is prepared exactly as explicit input would be; a ``None``
    default is written as ``None`` and not validated. The values are written back into
    ``args`` (the ADK calls the tool with this dict), the ADK's own validation must give
    them back exactly, and the policy judges a deep copy, so judged equals received. A
    validation failure is refused naming the parameter, without the SDK's text, which
    carries input values.
    """
    if not isinstance(tool, FunctionTool):
        return (
            None,
            f"only a google.adk.tools.FunctionTool can be judged, not {type(tool).__name__}",
        )
    why = _unstock_function_tool(tool)
    if why is not None:
        return None, why
    func = tool.func
    skip = tuple(getattr(tool, "_ignore_params", ()) or ())
    why = _unsupported_signature(func, skip)
    if why is not None:
        return None, f"its signature cannot be judged ({_with_remedy(func, why, skip)})"
    parameters = inspect.signature(func).parameters
    raw: dict[str, Any] = {}
    # A parameter whose default is ``None`` (``region: str = None``) and that the call
    # leaves out, or sets to ``null``, is judged as ``None`` and not validated against its
    # type (``None`` would fail ``str``, though the ADK runs the call). It is left for
    # Python to fill when every callable the call reaches that names it declares ``None``
    # too, so ``None`` is what runs (a ``validate_call`` function then keeps its own
    # unvalidated default). When one declares another default (a wrapper with a default of
    # its own) it is written as ``None`` explicitly, so that default never applies.
    none_defaults: list[str] = []
    for name, parameter in parameters.items():
        if name in skip:
            continue
        if name in args:
            if args[name] is None and parameter.default is None:
                none_defaults.append(name)
            continue
        if parameter.default is parameter.empty:
            return None, f"required parameter `{name}` was not supplied"
        if parameter.default is None:
            none_defaults.append(name)
            continue
        raw[name] = copy.deepcopy(parameter.default)
    unknown = sorted(k for k in args if k not in parameters)
    if unknown:
        return None, f"unknown parameter(s) {unknown!r}"
    # a parameter the ADK fills itself (``tool_context``, ``input_stream``) is not the
    # model's to set: the tool receives the ADK's value, or outside a live run the model's
    # unchecked one, and the policy would judge something else
    injected = sorted(k for k in args if k in skip)
    if injected:
        return None, f"the call sets {injected!r}, which the ADK fills itself"
    raw.update({k: v for k, v in args.items() if k not in none_defaults})
    converted, errors = _always_validated(tool, raw)
    if errors:
        # the parameter names, never the values (the ADK's own text carries the input)
        named = sorted({m.group(1) for e in errors if (m := _ERROR_PARAMETER.match(e))})
        which = f": {', '.join(f'`{n}`' for n in named)}" if named else ""
        return None, (
            f"{len(errors)} argument(s) do not validate against the tool's parameters{which}"
        )
    # ``run_async`` validates the dict the gate writes once more before the function runs.
    # Judge only arguments that are plain JSON data and that this validation gives back
    # exactly, type for type: whatever the validation does to them has then been applied,
    # and applying it again changes nothing
    explicit = _none_defaults_to_write(func, none_defaults, skip)
    written = {**converted, **{name: None for name in explicit}}
    again, errors = tool._preprocess_args_with_validation(copy.deepcopy(written))
    if errors:
        named = sorted({m.group(1) for e in errors if (m := _ERROR_PARAMETER.match(e))})
        if named and set(named) <= set(none_defaults):
            # only with the ADK's FUNCTION_TOOL_ARG_VALIDATION on, which checks every key
            return None, (
                "the ADK's own validation refuses None for "
                + ", ".join(f"`{n}`" for n in named)
                + "; declare the parameter Optional"
            )
    if errors or not _json_data(written) or not _identical(again, written):
        return None, (
            "the tool's validation does not give back the arguments it validated, so the "
            "gate cannot say what would run"
        )
    args.clear()
    args.update(written)
    judged = copy.deepcopy(written)
    judged.update({name: None for name in none_defaults})
    return judged, None


def _takes_skip(walk: Any) -> bool:
    """True if the kernel's ``reached_parameters`` takes the names a framework injects.

    ``recusal`` 0.15.0, the oldest this package allows, takes the callable alone; a later
    kernel also takes ``skip``. Handing ``skip`` to the former raises ``TypeError`` on
    every tool, which :func:`_none_defaults_to_write` would answer by writing every
    ``None`` default."""
    try:
        return "skip" in inspect.signature(walk).parameters
    except (TypeError, ValueError):
        return False


#: Decided once, at import, from the kernel that is installed.
_WALK_TAKES_SKIP = _takes_skip(_reached_parameters)


def _with_remedy(func: Any, why: str, skip: tuple = ()) -> str:
    """``why``, the kernel's reason a tool's signature cannot be judged, with what to do
    about it where the reason is one of the two the kernel's walk raises by name: a
    decorator that holds a second wrapper with a parameter of its own (``HiddenWrapper``)
    and a single-dispatch function that can run more than its base implementation
    (``DispatchedAtCallTime``). A kernel later than 0.15.0 raises them; beside 0.15.0
    neither shape is found, no handler here runs and ``why`` is returned as it is. The
    walk is made a second time only for a call that is refused already, and the remedy is
    added only when what it raises is, word for word, the reason the kernel gave."""
    try:
        _reached_parameters(func, skip) if _WALK_TAKES_SKIP else _reached_parameters(func)
    except _HiddenWrapper as refused:
        if str(refused) == why:
            return (
                f"{why}: a decorator on the tool holds a second wrapper of the function, in "
                "its closure or on the decorator object, which is not reached through "
                "`__wrapped__`; call the decorated function from a plain function and "
                "register that one"
            )
    except _DispatchedAtCallTime as refused:
        if str(refused) == why:
            return (
                f"{why}: it is a single-dispatch function (`functools.singledispatch` or "
                "`singledispatchmethod`) with an implementation registered beside its base "
                "one or in its place; call it from a plain function and register that one"
            )
    except Exception:  # noqa: BLE001, refused already, for the reason the kernel gave
        pass
    return why


def _none_defaults_to_write(func: Any, names: list[str], skip: tuple = ()) -> list[str]:
    """Of the parameters left at a ``None`` default, the ones that must be written as
    ``None``: those some callable the call reaches names with another default (or none),
    which would otherwise fill its own. The others are left for Python, which fills
    ``None`` on every level."""
    if not names:
        return []
    # called as the installed kernel takes it; typed loosely so that the call checks
    # beside a kernel of either signature
    walk: Callable[..., Any] = _reached_parameters
    try:
        levels = walk(func, skip) if _WALK_TAKES_SKIP else walk(func)
    except Exception:  # noqa: BLE001, read before by unsupported_signature; write them all
        return list(names)
    write = []
    for name in names:
        defaults = [level[name].default for level in levels if name in level]
        if any(default is not None for default in defaults):
            write.append(name)
    return write


#: How ``FunctionTool._validate_args`` names a failing parameter in its error text.
_ERROR_PARAMETER = re.compile(r"^Parameter '([^']+)'")


def _always_validated(tool: Any, raw: dict[str, Any]) -> tuple[dict[str, Any], list]:
    """The tool's own preparation with its type validation always on: what
    ``_preprocess_args_with_validation`` does when the ADK's ``FUNCTION_TOOL_ARG_VALIDATION``
    is enabled, built from ``FunctionTool``'s own methods.

    With the ADK's validation off (its default) the function would get the model's raw
    values, and a validating decorator on it (``pydantic.validate_call``) would convert
    them after judgment: ``"on"`` judged, ``True`` run. Converted here and written into
    the arguments, the values are already of the declared types, and every later
    validation (the ADK's, a decorator's) gives them back unchanged, which the fixed-point
    check below confirms for the ADK's.
    """
    preprocessed = tool._preprocess_args(raw)
    spec = tool._spec
    if not spec.has_signature:
        return preprocessed, []
    hints = spec.get_best_effort_type_hints()
    return tool._validate_args(
        preprocessed,
        ((name, hints.get(name, p.annotation)) for name, p in spec.signature.parameters.items()),
    )


def _agent_call(
    tool: Any, args: dict[str, Any], tool_context: Any = None
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """The call to an ``AgentTool`` as the sub-agent receives it, for judgment; or
    ``(None, reason)``.

    A stock ``AgentTool`` whose agent declares no ``input_schema`` hands the sub-agent one
    text: ``args["request"]``, or the whole dict as JSON. So the dict is the request, and the
    policy judges a detached copy of it; nothing the policy does reaches it.

    Judging the request is not enough: the sub-agent then makes tool calls of its own, so
    the call is judged only when those are covered too, either by the sub-agent's own gate
    (``gated_agent`` gates it) or by a ``RecusalPlugin`` the nested run inherits
    (``include_plugins``, the default, and the plugin among the runner's). Otherwise, with
    a gate attached by hand or the plugin and ``include_plugins=False``, the sub-agent's
    calls would run unjudged, so the call is refused. Refused too: a subclass (ADK's
    single-turn and task wrappers run the agent another way) and an agent with an
    ``input_schema``, whose own validation would decide what the sub-agent reads.
    """
    if type(tool) is not AgentTool:
        return None, f"{type(tool).__name__} runs its agent in a way this gate does not judge"
    if not is_gated(tool.agent) and not (
        tool.include_plugins
        and _runner_has_recusal_plugin(tool_context)
        and _unseen_reason(tool.agent) is None
    ):
        return None, (
            f"the tool calls agent {tool.agent.name!r} makes would not be judged: gate it "
            "with gated_agent, or let it inherit RecusalPlugin (include_plugins=True)"
        )
    # ADK's own reading (it looks into a workflow agent's first sub-agent), so the gate and
    # ``AgentTool.run_async`` cannot disagree about whether a schema applies
    if _agent_input_schema(tool.agent) is not None:
        return None, "an agent tool whose agent declares an input_schema is not supported"
    try:
        return _detached(args), None
    except TypeError as exc:
        return None, str(exc)


def _runner_has_recusal_plugin(tool_context: Any) -> bool:
    """Whether the runner driving this call carries a ``RecusalPlugin``, which a nested
    ``AgentTool`` run inherits when ``include_plugins`` is on."""
    invocation = getattr(tool_context, "_invocation_context", None)
    plugins = getattr(getattr(invocation, "plugin_manager", None), "plugins", None) or []
    return any(isinstance(plugin, RecusalPlugin) for plugin in plugins)


def _unstock_function_tool(tool: Any) -> Optional[str]:
    """Why this ``FunctionTool`` does not run a call the stock way, or None if it does."""
    name = type(tool).__name__
    for method in _STOCK_LIFECYCLE:
        if getattr(type(tool), method, None) is not getattr(FunctionTool, method):
            return f"{name} overrides FunctionTool.{method}, so its run is not the stock one"
        if method in getattr(tool, "__dict__", {}):
            return f"the tool object replaces FunctionTool.{method}"
    # every check here reads the class and the object's ``__dict__``; ADK reads
    # ``self.func`` and its methods through these, which a class of its own can answer with
    # anything
    for method in ("__getattribute__", "__getattr__"):
        if getattr(type(tool), method, None) is not getattr(FunctionTool, method, None):
            return f"{name} overrides `{method}`"
    # ADK reads the signature it filters and validates by from ``_spec``, built from the
    # function the tool was created with, and calls ``self.func``: they must be one
    spec = getattr(tool, "_spec", None)
    if getattr(spec, "func", None) is not tool.func:
        return "its function was replaced after the tool was built"
    return None


def _identical(a: Any, b: Any, depth: int = 0) -> bool:
    """Equal, and of the same type all the way down (``1 == True`` is not enough)."""
    if depth > 64 or type(a) is not type(b):
        return False
    if type(a) is list:
        return len(a) == len(b) and all(_identical(x, y, depth + 1) for x, y in zip(a, b))
    if type(a) is dict:
        return a.keys() == b.keys() and all(_identical(a[k], b[k], depth + 1) for k in a)
    return bool(a == b)


def _unjudgeable(tool_name: str, detail: str, proposal: Any, record: dict) -> str:
    """The reason for a call the gate refuses before any policy runs.

    Without recording arguments it is the plain text. With them
    (the audited gate) the refusal goes through ``decide`` like every other, so
    it is one entry on the log, bound to the arguments as the model proposed them, and the
    reason the model reads is the one on the record.
    """
    if not record:
        return f"Recusal refused `{tool_name}`: {detail}"
    bound = proposal if isinstance(proposal, dict) else {"input": proposal}

    def unjudgeable(_name: str, _input: dict[str, Any]) -> list[Finding]:
        return [Finding.fail("recusal_unjudgeable_input", severity="CRITICAL", message=detail)]

    try:
        _, reason = decide(tool_name, bound, unjudgeable, safe_errors=True, **record)
    except Exception as exc:  # noqa: BLE001, the refusal stands even if the record cannot be written
        return f"Recusal failed closed (policy error): RECUSAL_POLICY_ERROR ({type(exc).__name__})"
    return reason


def _record_for(recording: Optional[Recording], tool_context: Any) -> dict:
    """The recording arguments for this call, or none when the gate keeps no record."""
    return dict(recording(tool_context)) if recording is not None else {}


def _refusal(
    tool: Any, args: Any, policy: Policy, _tool_context: Any = None, **record: Any
) -> Optional[str]:
    """Return the reason a call is refused, or None to run it.

    The ADK hands the callback the same ``args`` dict the tool will be called with; the
    invocation is prepared in place (:func:`_prepared`) and the policy judges a detached
    copy of exactly what will run. A non-dict is refused: the tool would not accept it
    either, and the policy cannot read it.

    ``record`` goes straight to ``decide``, so a gate that puts the decision on a log
    (``audited_agent``, ``audited_plugin``) decides, and refuses, exactly as this one does. The
    entry for a judged call carries the fingerprint of the prepared invocation the tool
    is called with; a call refused before judgment is bound to the arguments as proposed
    (:func:`_prepared` changes ``args`` only when it succeeds).
    """
    tool_name = getattr(tool, "name", "?")
    if type(tool) is SetModelResponseTool:
        # the agent's structured answer, not an action: it runs nothing, and the model can
        # set the same answer as plain final text, which no callback sees. Judging the call
        # would guard nothing a text reply cannot also set, so no Recusal gate judges an
        # agent's answer (the README says so); judge what acts on it
        return None
    if not isinstance(args, dict):
        return _unjudgeable(tool_name, "arguments are not a mapping", args, record)
    try:
        if isinstance(tool, AgentTool):
            invocation, why = _agent_call(tool, args, _tool_context)
        else:
            invocation, why = _prepared(tool, args)
    except Exception as exc:  # noqa: BLE001, a preparation the gate cannot complete is refused
        return _unjudgeable(
            tool_name, f"the invocation cannot be prepared ({type(exc).__name__})", args, record
        )
    if invocation is None:
        return _unjudgeable(tool_name, str(why), args, record)
    try:
        decision, reason = decide(tool_name, invocation, policy, safe_errors=True, **record)
    except Exception as exc:  # noqa: BLE001, a buggy policy must not silently disable the gate
        return f"Recusal failed closed (policy error): RECUSAL_POLICY_ERROR ({type(exc).__name__})"
    return reason if decision == "deny" else None


def recusal_before_tool(
    policy: Policy, recording: Optional[Recording] = None
) -> BeforeToolCallback:
    """Return a ``before_tool_callback`` that judges each tool call before it runs.

    A refusal is returned as the tool's response, ``{"error": reason}``, so the model sees
    why on its next turn; ``None`` lets the ADK run the tool. ``recording`` puts every
    decision on an audit log (``recusal_google_adk.recording``).
    """

    def gate(
        tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
    ) -> Optional[dict[str, Any]]:
        try:
            record = _record_for(recording, tool_context)
        except Exception as exc:  # noqa: BLE001, a gate that cannot record must not run the call
            return {
                "error": "Recusal failed closed (policy error): "
                f"RECUSAL_POLICY_ERROR ({type(exc).__name__})"
            }
        reason = _refusal(tool, args, policy, tool_context, **record)
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

    def __init__(
        self, policy: Policy, name: str = "recusal", recording: Optional[Recording] = None
    ):
        super().__init__(name=name)
        self._policy = policy
        self._recording = recording

    async def before_run_callback(self, *, invocation_context: Any) -> Optional[Any]:
        """Refuse the whole run when any agent the runner drives can execute something no
        callback sees (``_unseen_reason``): a plugin is handed tool calls only, so model-
        written code, a provider-side tool or a task delegation would otherwise run
        unjudged. Returning content ends the run before any agent starts. A nested run (an
        ``AgentTool`` that inherits the plugin) is checked the same way."""
        from google.genai import types

        try:
            reason = _run_unseen_reason(invocation_context)
        except Exception as exc:  # noqa: BLE001, a tree the gate cannot read is refused
            reason = f"the agents cannot be checked ({type(exc).__name__})"
        if reason is None:
            return None
        return types.Content(
            role="model",
            parts=[types.Part(text=f"Recusal refused this run: {reason}")],
        )

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> Optional[dict[str, Any]]:
        try:
            record = _record_for(self._recording, tool_context)
        except Exception as exc:  # noqa: BLE001, a gate that cannot record must not run the call
            return {
                "error": "Recusal failed closed (policy error): "
                f"RECUSAL_POLICY_ERROR ({type(exc).__name__})"
            }
        reason = _ordering_reason(self, tool_context)
        if reason is not None:
            return {"error": _unjudgeable(tool.name, reason, tool_args, record)}
        reason = _refusal(tool, tool_args, self._policy, tool_context, **record)
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
    """Whether the agent's LAST before-tool callback is a Recusal gate: a callback after it
    could change the arguments the gate judged (``gated_agent`` always appends it last)."""
    callbacks = _callbacks(agent)
    return bool(callbacks) and getattr(callbacks[-1], "recusal_gate", False) is True


def _run_unseen_reason(invocation_context: Any) -> Optional[str]:
    """Why a run can execute something no callback sees, or None.

    The tree is walked from its ROOT: on a continued session the ADK starts the run at the
    agent that replied last, often a sub-agent, and the run can still transfer anywhere in
    the tree. The run's own settings are read too: ``RunConfig(support_cfc=True)`` gives the
    root agent the model provider's code execution, which no callback sees.
    """
    run_config = getattr(invocation_context, "run_config", None)
    if getattr(run_config, "support_cfc", False):
        return (
            "the run sets support_cfc, which gives the agent the model provider's code "
            "execution; no callback sees it"
        )
    agent = getattr(invocation_context, "agent", None)
    root = getattr(agent, "root_agent", None) or agent
    return _unseen_reason(root)


def _unseen_reason(agent: Any, _seen: Optional[set] = None) -> Optional[str]:
    """Why some agent in this tree can run something no Recusal callback sees, or None.

    The rules ``gated_agent`` applies when it builds, over the whole tree: every agent,
    its ``sub_agents`` and the agent behind every ``AgentTool`` (including the tools the
    ADK adds for ``task`` and ``single_turn`` sub-agents, which it refuses). A plugin sees
    only tool calls, so ``RecusalPlugin`` refuses a run whose tree fails this rather than
    let a ``code_executor``, a provider-side tool or a task delegation run unjudged.
    """
    seen = set() if _seen is None else _seen
    if id(agent) in seen:
        return None
    seen.add(id(agent))
    reason = _ungateable_reason(agent)
    if reason is not None:
        return f"agent {getattr(agent, 'name', '?')!r}: {reason}"
    if isinstance(agent, LlmAgent):
        for tool in agent.tools:
            if isinstance(tool, AgentTool):
                reason = _unseen_reason(tool.agent, seen)
                if reason is not None:
                    return reason
    for sub in getattr(agent, "sub_agents", ()) or ():
        reason = _unseen_reason(sub, seen)
        if reason is not None:
            return reason
    return None


def _ungateable_reason(agent: Any) -> Optional[str]:
    """Why an agent can run something the callback never sees, or None if it cannot."""
    # the exact classes: a subclass of a workflow agent can run anything of its own
    if type(agent) in _WORKFLOW_AGENTS:
        return None
    if type(agent) is not LlmAgent:
        return (
            f"{type(agent).__name__} is not a plain LlmAgent; a subclass can run tools or "
            "code the before-tool callback never sees"
        )
    if agent.code_executor is not None:
        return "it has a code_executor; model-written code runs outside the tool path"
    config = getattr(agent, "generate_content_config", None)
    if getattr(config, "tools", None):
        return (
            "its generate_content_config declares tools, which the model provider runs "
            "itself; no before-tool callback sees them"
        )
    why = _model_provider_tools(agent.model)
    if why is not None:
        return why
    for tool in agent.tools:
        why = _unjudged_tool_kind(tool)
        if why is not None:
            return why
    return None


#: The options a model wrapper hands its provider that turn on a tool the provider runs
#: itself, as verified in ADK 2.14's ``LiteLlm``: ``web_search_options`` turns on the
#: provider's web search. ``tools`` is not one (``LiteLlm`` drops it and sends the agent's
#: own), nor is ``tool_choice`` (it steers the agent's own tools). Any other provider option
#: (temperature, timeouts, ...) is configuration the gate does not read.
_PROVIDER_TOOL_OPTIONS = frozenset({"web_search_options"})


def _provider_tool_option(options: Any, depth: int = 0) -> Optional[str]:
    """The provider-tool option found anywhere in a model's options, or None: at the top
    level or nested (``extra_body={"web_search_options": ...}`` reaches the provider the
    same way), and an OpenRouter web plugin (``extra_body={"plugins": [{"id": "web"}]}``)."""
    if depth > 16:
        return "options nested too deep to read"
    if isinstance(options, dict):
        for key, value in options.items():
            if key in _PROVIDER_TOOL_OPTIONS:
                return f"`{key}`"
            if key == "plugins" and isinstance(value, (list, tuple)):
                if any(isinstance(p, dict) and p.get("id") == "web" for p in value):
                    return "an OpenRouter web plugin"
            found = _provider_tool_option(value, depth + 1)
            if found is not None:
                return found
    elif isinstance(options, (list, tuple)):
        for value in options:
            found = _provider_tool_option(value, depth + 1)
            if found is not None:
                return found
    return None


def _model_provider_tools(model: Any, depth: int = 0) -> Optional[str]:
    """Why the agent's model object enables tools the provider runs itself, or None. A
    model wrapper that passes extra options to the provider (``LiteLlm`` keeps them in
    ``_additional_args``) is read for the options that declare tools; a ``FallbackModel``'s
    every model is read."""
    if depth > 8 or model is None or isinstance(model, str):
        return None
    extra = getattr(model, "_additional_args", None)
    if isinstance(extra, dict):
        found = _provider_tool_option(extra)
        if found is not None:
            return (
                f"its model passes {found} to the provider, which turns on a tool the "
                "provider runs itself; no before-tool callback sees it"
            )
    for inner in getattr(model, "models", None) or ():
        why = _model_provider_tools(inner, depth + 1)
        if why is not None:
            return why
    return None


#: ADK's own tools that run locally and hide nothing from the callbacks, accepted on an
#: agent by exact class: ``preload_memory`` only adds memory text to the model request and
#: is never called; ``load_artifacts`` is called through the callbacks like any tool, where
#: the gate, which cannot judge it, refuses the call (as before this release).
_LOCAL_ADK_TOOLS = (PreloadMemoryTool, LoadArtifactsTool)


def _unjudged_tool_kind(tool: Any) -> Optional[str]:
    """Why a tool on an agent's list never reaches the before-tool callback as a call the
    gate can judge, or None if it does. Default-deny: a function (the ADK wraps it in a
    ``FunctionTool``), a ``FunctionTool``, and a plain ``AgentTool`` pass; anything else is
    refused, so a tool kind added in a later ADK release is refused too.

    The kinds this refuses: the built-in tools the model provider runs on its own side
    (``google_search``, ``url_context``, ``VertexAiSearchTool``, ...), which edit the model
    request and never reach a callback; toolsets (MCP, OpenAPI, ...), whose tools the gate
    could not judge; and an ``AgentTool`` subclass.
    """
    if isinstance(tool, AgentTool):
        if type(tool) is not AgentTool:
            return f"{type(tool).__name__} is an AgentTool subclass, which it cannot gate"
        return None
    if isinstance(tool, FunctionTool):
        # ``process_llm_request`` edits the request sent to the model; the ADK's own tools
        # that override it add instructions, but an override could add a provider-side tool
        if "process_llm_request" in getattr(tool, "__dict__", {}):
            return "the tool object replaces process_llm_request"
        method = type(tool).process_llm_request
        if method is not BaseTool.process_llm_request and not getattr(
            method, "__module__", ""
        ).startswith("google.adk."):
            return (
                f"{type(tool).__name__} overrides process_llm_request, which can add tools "
                "the model provider runs itself"
            )
        return None
    if type(tool) in _LOCAL_ADK_TOOLS and "process_llm_request" not in getattr(
        tool, "__dict__", {}
    ):
        return None
    if isinstance(tool, (BaseTool, BaseToolset)):
        name = getattr(tool, "name", type(tool).__name__)
        if type(tool) is FinishTaskTool:
            return (
                f"tool {name!r} is how an agent in task mode finishes its task, which this "
                "gate cannot judge; task mode is not supported"
            )
        return (
            f"tool {name!r} ({type(tool).__name__}) is not a function, a FunctionTool or a "
            "plain AgentTool, so the gate cannot judge it (a built-in tool the model "
            "provider runs, a toolset, or another tool kind)"
        )
    if callable(tool):
        return None
    return f"{tool!r} is not a tool"


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


def gated_agent(
    agent: BaseAgent, policy: Policy, recording: Optional[Recording] = None
) -> BaseAgent:
    """Return a copy of the agent whose every tool call is judged by Recusal.

    The gate is appended as the LAST before-tool callback; ``sub_agents`` and the agent
    behind every ``AgentTool`` are gated the same way. Raises ValueError if the agent, or
    any agent under it, can run something the callback cannot see. ``recording`` puts
    every decision of every gate it installs on an audit log.
    """
    reason = _ungateable_reason(agent)
    if reason is not None:
        raise ValueError(f"Recusal cannot gate agent {agent.name!r}: {reason}")
    update: dict[str, Any] = {
        "sub_agents": [gated_agent(sub, policy, recording) for sub in agent.sub_agents]
    }
    if isinstance(agent, LlmAgent):
        tools: list[Any] = []
        for tool in agent.tools:
            if isinstance(tool, AgentTool):
                # every setting the tool was built with, not only the agent: a dropped
                # ``include_plugins`` would change which plugins the sub-agent runs under
                tools.append(
                    AgentTool(
                        gated_agent(tool.agent, policy, recording),
                        skip_summarization=tool.skip_summarization,
                        include_plugins=tool.include_plugins,
                        propagate_grounding_metadata=tool.propagate_grounding_metadata,
                    )
                )
            else:
                tools.append(tool)
        update["tools"] = tools
        update["before_tool_callback"] = _callbacks(agent) + [
            recusal_before_tool(policy, recording)
        ]
    return agent.clone(update=update)
