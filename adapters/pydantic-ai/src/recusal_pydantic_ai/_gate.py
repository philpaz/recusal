"""The gate: one refusal function, and the two ways to attach it to a Pydantic AI agent.

The hook is Pydantic AI's own. An ``Agent`` runs each capability's ``before_tool_execute``
hook after a tool call's arguments are validated and immediately before the tool runs, with
the call, the tool definition and the validated argument dict the tool will receive. The
hook returns the arguments to run with, or raises ``ToolFailed`` to answer the call with a
failed tool result instead of running it. ``RecusalGate(policy)`` judges the call with
``recusal.decide`` and, on a refusal, raises ``ToolFailed(reason)``, so the model reads
Recusal's reason on its next turn, the tool never runs, and no retry budget is spent (a
refusal is not a model mistake to correct). Every tool call in a model turn is judged
separately. Every toolset the agent runs, ``tools=``, ``toolsets=`` or a capability's own
tools, executes through the same ``ToolManager``, so the hook sees each call by name and
validated arguments. Output tools skip that hook; ``RecusalGate`` judges an output function
(``output_type=fn``) in ``before_output_process`` before it runs, and a call handed to the
application to run (an ``ExternalToolset``) in ``handle_deferred_tool_calls``, where it
then calls the application's own handler (``deferred=``) with the calls it allowed.
Another capability that answers deferred calls is offered them before the gate, so a run
with one is refused. A ``Choices`` pick, whose value Pydantic AI calls when it is
callable, is judged as ``choice``. A toolset runs after every hook, so a run holding an
application toolset that runs other tools' calls itself has every call refused
(``_rewriting_wrapper`` says which): the run's toolset is read at the time of each call,
and when the run is assembled (``get_wrapper_toolset``). A request offering the model a
tool the provider runs is refused (``prepare_model_request``).
The policy is given the name the model calls the tool by: prefixed or renamed, if a
``PrefixedToolset`` or ``RenamedToolset`` says so.

Supported arguments are data-only: JSON scalars, lists and string-keyed dicts. Containers
are detached by construction. Models, dataclasses, Enum members and other objects are
refused before policy evaluation. Declare data-only parameters (a ``str`` or ``int`` in
place of an Enum) and resolve application types inside the tool body. No application copy
method or serializer is called.

What the hook cannot see:

- an agent another agent calls from inside a tool function: its tool calls run under its
  own capabilities, not the caller's. Give every agent its own gate, at construction or
  per run inside the delegating tool; ``is_gated(agent)`` reads only the agent it is given.
- a tool call whose arguments fail validation: Pydantic AI answers it with a retry prompt
  before any execute hook runs, and the tool does not run either.
"""

from __future__ import annotations

import inspect
from collections import deque
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults
from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    CombinedCapability,
    Hooks,
    WrapperCapability,
)
from pydantic_ai.capabilities.combined import is_deferred_group
from pydantic_ai.exceptions import ModelRetry, ToolFailed, UserError
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.native_tools._tool_search import ToolSearchTool
from pydantic_ai.output import OutputContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.set_metadata import SetMetadataToolset

from recusal import Finding, decide
from recusal.invocation import DeclaredSignature as _DeclaredSignature
from recusal.invocation import detached as _detached
from recusal.invocation import json_data as _json_data
from recusal.invocation import reached_parameters as _reached_parameters

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.capabilities.hooks import BeforeToolExecuteHookFunc
    from pydantic_ai.models import ModelRequestContext

#: A Recusal policy: ``policy(tool_name, tool_input)`` returns a list of findings.
Policy = Callable[[str, dict[str, Any]], list[Finding]]

#: A deferred-call handler, as ``HandleDeferredToolCalls`` takes one: ``handler(ctx,
#: requests)`` returns the results it has, or None. Sync or async.
DeferredHandler = Callable[
    [Any, DeferredToolRequests],
    Union[Optional[DeferredToolResults], Awaitable[Optional[DeferredToolResults]]],
]


def _unjudgeable(tool_name: str, detail: str, proposal: Any, record: dict[str, Any]) -> str:
    """The reason for a call the gate refuses before any policy runs.

    Without recording arguments it is the plain text. With them (the audited gate) the
    refusal goes through ``decide`` like every other, so it is one entry on the log, bound
    to the call as the model proposed it, and the reason the model reads is the one on the
    record.
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


def _refusal(
    tool_name: str, args: Any, policy: Policy, *, proposal: Any = None, **record: Any
) -> Optional[str]:
    """Return the reason a call is refused, or None to run it.

    Pydantic AI hands the hook the validated argument dict the tool will be called with,
    and the callers complete it first with any default the function would fill in itself
    (``_with_defaults``), so the policy judges what would run. A non-dict is refused: the
    tool would not accept it either, and the policy cannot read it.

    The policy judges detached data, built by ``_detached``, and the tool runs with the
    validated dict itself. Unsupported values refuse before the policy is called. No
    model projection, application copy method or serializer participates in the decision.

    ``record`` goes straight to ``decide`` (``audit=``, ``surface=``, ``tool_use_id=``, ...),
    so the audited gate decides, and refuses, exactly as the plain one does: one function,
    no second copy to drift. The entry for a call that runs carries the fingerprint of the
    detached arguments, which are equal to the dict the tool receives. ``proposal`` is the
    call as the model sent it, and is what the entry for a call refused before judgment is
    bound to.
    """
    if not isinstance(args, dict):
        return _unjudgeable(
            tool_name, "arguments are not a mapping", args if proposal is None else proposal, record
        )
    try:
        observed = _detached(args)
    except Exception as exc:  # noqa: BLE001, an unsupported observation must not run
        return _unjudgeable(
            tool_name,
            f"arguments cannot be detached for judging ({type(exc).__name__}); "
            "declare data-only parameters and validate inside the tool body",
            proposal,
            record,
        )
    try:
        decision, reason = decide(tool_name, observed, policy, safe_errors=True, **record)
    except Exception as exc:  # noqa: BLE001, a buggy policy must not silently disable the gate
        return f"Recusal failed closed (policy error): RECUSAL_POLICY_ERROR ({type(exc).__name__})"
    return reason if decision == "deny" else None


def recusal_before_tool_execute(policy: Policy) -> BeforeToolExecuteHookFunc:
    """Return a ``before_tool_execute`` hook that judges each tool call before it runs.

    For ``Hooks(before_tool_execute=...)``. A refusal is raised as ``ToolFailed(reason)``,
    which Pydantic AI turns into a failed tool result the model reads on its next turn;
    the returned arguments run the tool unchanged.

    ``Hooks`` declares no ordering, so list this form after every other capability, and
    use it only where no ``innermost`` capability is added: one runs after it wherever it
    is listed, and a hook that runs after it can change the arguments it judged. It judges
    tool calls only; ``RecusalGate`` also judges output functions and calls handed to the
    application, and refuses a run whose toolsets could rewrite a call. Prefer it.
    """

    async def gate(
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        args, unseen = _with_defaults(ctx, call.tool_name, args)
        if unseen:
            raise ToolFailed(_unjudgeable(call.tool_name, unseen, None, {}))
        reason = _refusal(call.tool_name, args, policy)
        if reason is not None:
            raise ToolFailed(reason)
        return args

    gate.recusal_gate = True  # type: ignore[attr-defined]
    return gate


@dataclass
class RecusalGate(AbstractCapability[Any]):
    """The gate as a capability: ``Agent(..., capabilities=[RecusalGate(policy)])``.

    Declares the ``innermost`` position, so Pydantic AI sorts it after every ordinary
    capability however the list is written, and it judges the arguments as they reach the
    tool. List it directly and last: a call is refused when a capability with a
    ``before_tool_execute`` of the application's is dispatched after the gate (another
    ``innermost`` capability listed after it, or anything listed after a gate placed
    inside a ``WrapperCapability``, which does not pass its position on).

    ``policy`` is required. A gate has no default policy: what to refuse is the adopter's
    decision, never the adapter's.

    ``deferred`` is the application's handler for deferred calls (approvals, and calls it
    executes itself), the function ``HandleDeferredToolCalls`` would take. Pydantic AI
    offers deferred calls to each capability in turn and the innermost gate comes last, so
    a handler in any other capability would be handed calls no policy has judged; a run
    with one is refused. Given here, the handler is called by the gate, after the gate has
    judged the calls, with the ones it allowed.

    A gate cannot be loaded on demand (``defer_loading``): Pydantic AI runs no hook of a
    capability that is not loaded, so the gate would not judge anything.
    """

    policy: Policy
    deferred: Optional[DeferredHandler] = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.defer_loading:
            raise UserError(
                "A Recusal gate cannot be loaded on demand (`defer_loading=True`): Pydantic AI "
                "runs no hook of a capability that is not loaded, so it would judge nothing."
            )

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")

    # --- what the audited gate records; the plain gate records nothing -------------------

    def _record(self, tool_call_id: Optional[str], surface: str) -> dict[str, Any]:
        """The recording arguments ``decide`` takes for one decision (none here)."""
        return {}

    def _proposal(self, call: Optional[ToolCallPart]) -> Any:
        """The call as the model sent it, for a call refused before judgment (none here)."""
        return None

    # --- tool calls ----------------------------------------------------------------------

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        record = self._record(call.tool_call_id, SURFACE_TOOL)
        unseen = (tool_def.metadata or {}).get(_RUN_REFUSAL)
        if not unseen:
            # The run's toolset as it is now, with every capability's wrapper and each dynamic
            # toolset's current contents: a wrapper added outside the gate's own is here.
            toolset = getattr(getattr(ctx, "tool_manager", None), "toolset", None)
            unseen = _rewriting_wrapper(toolset) if toolset is not None else None
        if not unseen:
            unseen = _rewriter_after(getattr(ctx, "root_capability", None), self)
        if not unseen:
            args, unseen = _with_defaults(ctx, call.tool_name, args)
        if unseen:
            raise ToolFailed(_unjudgeable(call.tool_name, unseen, self._proposal(call), record))
        reason = _refusal(
            call.tool_name, args, self.policy, proposal=self._proposal(call), **record
        )
        if reason is not None:
            raise ToolFailed(reason)
        return args

    async def before_run(self, ctx: RunContext[Any]) -> None:
        """Refuse, before anything runs, a run the gate could not hold to its policy.

        - A realtime session: it calls none of the request and output hooks the gate judges
          in, and hands its tools straight to the provider.
        - Another capability that answers deferred calls (see ``deferred``). Checked here
          as well as before each request, because a run resumed from a history that ends in
          a tool call executes it before any request is made."""
        if getattr(ctx, "realtime", False):
            raise UserError(
                "Recusal refused the run: a realtime session calls none of the request and "
                "output hooks the gate judges in, so it cannot be gated."
            )
        _refuse_deferred_handlers_elsewhere(ctx)

    async def prepare_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Refuse a request the gate could not hold to its policy, before it is sent. Read
        before each attempt, once every capability has shaped the request.

        - A tool the provider runs (web search, code execution, web fetch, a provider-side
          MCP server, ...): it runs with arguments the model chose where no hook of the
          application sees them. The model itself is asked which native tools the request
          will carry, so one it drops, or replaces with its local version, is not counted.
          Provider-side tool search is allowed: it finds tools, and the calls to them are
          judged as any other.
        - Another capability that answers deferred calls: it is offered them before the
          gate (see ``deferred``)."""
        _refuse_deferred_handlers_elsewhere(ctx)
        provider_run = _provider_run_tools(request_context)
        if provider_run:
            raise UserError(
                f"Recusal refused the model request: it offers tools the provider runs "
                f"({', '.join(provider_run)}), and their calls cannot be judged before they "
                "run. Remove them from the gated agent; where a tool has a local version, "
                "turn its provider side off (`native=False`)."
            )
        return request_context

    def get_wrapper_toolset(self, toolset: AbstractToolset[Any]) -> Optional[AbstractToolset[Any]]:
        """Read the run's assembled toolset. The execute hooks run before the toolset does,
        so a toolset that wraps another and runs its calls itself (``call_tool``) can change
        the arguments after the gate judged them. When one is in the tree, every tool of the
        run is marked, and ``before_tool_execute`` refuses each call, on the record. The
        wrappers Pydantic AI ships (prefixing, renaming, filtering, approval, tool search,
        durable execution) route or re-validate a call and do not change its arguments.
        ``before_tool_execute`` also reads the run's toolset at the time of each call."""
        reason = _rewriting_wrapper(toolset)
        if reason is None:
            return None
        return SetMetadataToolset(toolset, metadata={_RUN_REFUSAL: reason})

    # --- output functions ----------------------------------------------------------------

    async def before_output_process(
        self,
        ctx: RunContext[Any],
        *,
        output_context: OutputContext,
        output: Any,
    ) -> Any:
        """Judge an output function before it runs: ``output_type=fn`` runs the
        application's function with values the model chose, so it is a call like any tool's.
        A multi-argument function's arguments are judged as they are (a dict); a
        single-argument function's value as ``{"output": value}``, under the function's
        name. Plain structured output with no function runs nothing and is not judged. A
        refusal asks the model again (``ModelRetry``); the function does not run."""
        if _is_choices(output_context.output_type):
            return self._judge_choice(ctx, output_context, output)
        if not output_context.has_function:
            return output
        call = output_context.tool_call
        record = self._record(call.tool_call_id if call is not None else None, SURFACE_OUTPUT)
        name = output_context.function_name
        self._refuse_output_changed_after(ctx, name or "output", output, record)
        if name is None:
            raise ModelRetry(
                _unjudgeable(
                    "output",
                    "the output function that would run is not known: a union of output "
                    "functions under NativeOutput or PromptedOutput does not name the one it "
                    "will call; use tool output, where each function is its own output tool",
                    output,
                    record,
                )
            )
        args = output if isinstance(output, dict) else {"output": output}
        reason = _refusal(name, args, self.policy, proposal=self._proposal(call), **record)
        if reason is not None:
            raise ModelRetry(reason)
        return output

    def _refuse_output_changed_after(
        self, ctx: RunContext[Any], name: str, output: Any, record: dict[str, Any]
    ) -> None:
        """Refuse an output a ``before_output_process`` of the application's, dispatched
        after the gate, could change before the function or action runs."""
        reason = _rewriter_after(
            getattr(ctx, "root_capability", None), self, "before_output_process"
        )
        if reason is not None:
            raise ModelRetry(_unjudgeable(name, reason, output, record))

    def _judge_choice(
        self, ctx: RunContext[Any], output_context: OutputContext, output: Any
    ) -> Any:
        """Judge a ``Choices`` pick, whose value Pydantic AI calls when it is callable (an
        action). The hook is handed the picked key, not the values, so every final pick is
        judged, as ``choice`` with ``{"choice": key, "set": name}``, the name being the
        set's (``Choices(..., name=...)``, ``"Choices"`` by default). A partial pick runs
        nothing."""
        if getattr(ctx, "partial_output", False):
            return output
        call = output_context.tool_call
        record = self._record(call.tool_call_id if call is not None else None, SURFACE_OUTPUT)
        self._refuse_output_changed_after(ctx, "choice", output, record)
        name = getattr(output_context.output_type, "__name__", "Choices")
        picked = {"choice": output, "set": name}
        reason = _refusal("choice", picked, self.policy, proposal=self._proposal(call), **record)
        if reason is not None:
            raise ModelRetry(reason)
        return output

    # --- calls answered outside the run --------------------------------------------------

    async def handle_deferred_tool_calls(
        self,
        ctx: RunContext[Any],
        *,
        requests: DeferredToolRequests,
    ) -> Optional[DeferredToolResults]:
        """Judge the calls the run hands to the application to execute (an
        ``ExternalToolset``, a tool that raises ``CallDeferred``) before they leave it, as
        the model sent them, which is what the application receives: a refused call is
        answered with the refusal, which the model reads. Then ``deferred``, if given, is
        called with what is left; a result it returns for a refused call is dropped. Calls
        waiting for approval are judged when they run, by ``before_tool_execute``."""
        refused: dict[str, ToolFailed] = {}
        for call in requests.calls:
            try:
                args: Any = call.args_as_dict(raise_if_invalid=True)
            except Exception:  # noqa: BLE001, arguments that are not an object are refused
                args = None
            reason = _refusal(
                call.tool_name,
                args,
                self.policy,
                proposal=self._proposal(call),
                **self._record(call.tool_call_id, SURFACE_DEFERRED),
            )
            if reason is not None:
                refused[call.tool_call_id] = ToolFailed(reason)
        results = DeferredToolResults()
        if self.deferred is not None:
            left = (
                requests.remaining(DeferredToolResults(calls=dict(refused)))
                if refused
                else requests
            )
            if left is not None:
                answer = self.deferred(ctx, left)
                if inspect.isawaitable(answer):
                    answer = await answer
                if answer is not None:
                    results.update(answer)
        results.calls.update(refused)
        return results if results.calls or results.approvals else None


#: The audit surfaces, one per hook that adjudicates.
SURFACE_TOOL = "pydantic_ai.before_tool_execute"
SURFACE_OUTPUT = "pydantic_ai.before_output_process"
SURFACE_DEFERRED = "pydantic_ai.handle_deferred_tool_calls"

#: The tool metadata key ``get_wrapper_toolset`` sets on every tool of a run it cannot see
#: all of; its value is the reason.
_RUN_REFUSAL = "recusal_unjudgeable_run"


def _is_choices(output_type: Any) -> bool:
    """True for the key class a ``Choices(...)`` call builds, which is what the output hook
    is handed for a ``Choices`` output. ``Choices`` is a factory: each call makes a new class
    in ``pydantic_ai.output`` that builds its own Pydantic schema. ``StructuredDict`` makes
    the only other such class there, and that one is a ``dict``."""
    return (
        isinstance(output_type, type)
        and output_type.__module__ == "pydantic_ai.output"
        and "__get_pydantic_core_schema__" in vars(output_type)
        and not issubclass(output_type, dict)
    )


def _deferred_handlers_elsewhere(capability: Any) -> list[str]:
    """The capabilities in this tree, other than a gate, that answer deferred calls.

    A ``WrapperCapability`` that only passes the question on (a capability function is
    one) is read through to what it wraps. A ``Hooks`` counts when a deferred hook is
    registered on it."""
    if capability is None or isinstance(capability, RecusalGate):
        return []
    if isinstance(capability, CombinedCapability):
        return [
            name
            for child in capability.capabilities
            for name in _deferred_handlers_elsewhere(child)
        ]
    own = type(capability).handle_deferred_tool_calls
    if isinstance(capability, WrapperCapability):
        if own is WrapperCapability.handle_deferred_tool_calls:
            return _deferred_handlers_elsewhere(capability.wrapped)
    elif isinstance(capability, Hooks):
        if own is not Hooks.handle_deferred_tool_calls:
            return [type(capability).__name__]
        try:
            registered = bool(capability._get("handle_deferred_tool_calls"))
        except Exception:  # noqa: BLE001, a Hooks this cannot read may answer them
            registered = True
        return ["Hooks"] if registered else []
    if own is AbstractCapability.handle_deferred_tool_calls:
        return []
    return [type(capability).__name__]


def _refuse_deferred_handlers_elsewhere(ctx: Any) -> None:
    elsewhere = _deferred_handlers_elsewhere(getattr(ctx, "root_capability", None))
    if elsewhere:
        raise UserError(
            f"Recusal refused the run: {', '.join(elsewhere)} answers deferred tool calls, "
            "and Pydantic AI offers them to it before the gate, so it would be handed calls "
            "no policy has judged. Give its handler to the gate instead: "
            "`RecusalGate(policy, deferred=handler)`."
        )


def _holds(capability: Any, gate: Any, depth: int = 0) -> bool:
    """True if this capability is the gate, or a wrapper or group around it."""
    if capability is gate:
        return True
    if depth > 32:
        return False
    if isinstance(capability, WrapperCapability):
        return _holds(capability.wrapped, gate, depth + 1)
    if isinstance(capability, CombinedCapability):
        return any(_holds(child, gate, depth + 1) for child in capability.capabilities)
    return False


def _changes_arguments(capability: Any, gate: Any, hook: str, depth: int = 0) -> Optional[str]:
    """The name of a capability here whose ``hook`` is the application's and so could change
    what the gate judged, or None. The gate itself, Pydantic AI's own capabilities and
    wrappers that only pass the hook on are not such."""
    if capability is gate or isinstance(capability, RecusalGate) or depth > 32:
        return None
    if isinstance(capability, CombinedCapability):
        for child in capability.capabilities:
            name = _changes_arguments(child, gate, hook, depth + 1)
            if name is not None:
                return name
        return None
    own = getattr(type(capability), hook)
    if isinstance(capability, Hooks):
        if own is not getattr(Hooks, hook):
            return type(capability).__name__
        try:
            registered = bool(capability._get(hook))
        except Exception:  # noqa: BLE001, a Hooks this cannot read may have one
            registered = True
        return "Hooks" if registered else None
    if isinstance(capability, WrapperCapability):
        if _stock(own):
            return _changes_arguments(capability.wrapped, gate, hook, depth + 1)
        return type(capability).__name__
    return None if _stock(own) else type(capability).__name__


def _rewriter_after(
    node: Any, gate: Any, hook: str = "before_tool_execute", depth: int = 0
) -> Optional[str]:
    """Why a capability of this run could change what the gate judged after it judged, or
    None. Pydantic AI runs ``hook`` in dispatch order, so the gate must be the last
    capability with one of the application's: it is, when listed directly, by its
    ``innermost`` position, unless another ``innermost`` capability is listed after it;
    inside a ``WrapperCapability`` it has no position of its own among the wrapper's
    neighbours. Read level by level: within each group, what is dispatched after the member
    that holds the gate; and each wrapper around the gate, for a hook of its own."""
    if node is None or node is gate:
        return None
    unknown = "the gate is not among the capabilities this run dispatches, so its place is unknown"
    if depth > 32:
        return unknown
    if isinstance(node, CombinedCapability):
        members = list(getattr(node, "_dispatch", None) or node.capabilities)
        place = next((i for i, member in enumerate(members) if _holds(member, gate)), None)
        if place is None:
            return unknown
        reason = _rewriter_after(members[place], gate, hook, depth + 1)
        if reason is not None:
            return reason
        for member in members[place + 1 :]:
            name = _changes_arguments(member, gate, hook)
            if name is not None:
                return (
                    f"capability {name} has a `{hook}` that runs after the gate, so it could "
                    "change what the gate judged; list the gate directly and last"
                )
        return None
    if isinstance(node, WrapperCapability) and _holds(node, gate):
        if not _stock(getattr(type(node), hook)):
            return (
                f"capability {type(node).__name__}, which holds the gate, has a `{hook}` of "
                "its own, so it could change what the gate judged"
            )
        return _rewriter_after(node.wrapped, gate, hook, depth + 1)
    return unknown


def _function_of(ctx: Any, tool_name: str) -> tuple[Any, bool, bool]:
    """The Python function a function tool runs, whether it takes the run context first,
    and whether Pydantic AI validates its arguments; ``(None, False, True)`` for a tool that
    runs no local function Pydantic AI built a schema for (an MCP tool, a toolset of the
    application's own)."""
    tools = getattr(getattr(ctx, "tool_manager", None), "tools", None)
    tool = tools.get(tool_name) if isinstance(tools, Mapping) else None
    for _ in range(32):
        # a combined toolset hands out a record that points at the tool it came from
        source = getattr(tool, "source_tool", None)
        if getattr(tool, "call_func", None) is not None or source is None:
            break
        tool = source
    schema = getattr(getattr(tool, "call_func", None), "__self__", None)
    function = getattr(schema, "function", None)
    if not callable(function):
        return None, False, True
    return function, bool(getattr(schema, "takes_ctx", False)), _validates(schema)


def _validates(schema: Any) -> bool:
    """False for a function schema that accepts anything: ``Tool.from_schema`` gives the
    function the "any" validator, which hands back whatever it is given, so the arguments
    reach the function as the model sent them."""
    probe = object()
    try:
        return schema.validator.validate_python(probe) is not probe
    except Exception:  # noqa: BLE001, a validator that refuses an arbitrary object validates
        return True


def _variadic(parameter: Any) -> bool:
    return parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)


def _only_forwards(level: Any) -> bool:
    """True for a callable whose only parameters are ``*args`` and ``**kwargs``: it binds
    nothing itself, so everything it is called with goes on to what it wraps."""
    return bool(level) and all(_variadic(parameter) for parameter in level.values())


def _same_value(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        return type(left) is type(right) and bool(left == right)
    except Exception:  # noqa: BLE001, a comparison that raises is not a match
        return False


def _opaque(level: Any, inner: Any) -> Optional[str]:
    """The parameter that keeps a wrapper from being transparent, or None if it is.

    Transparent means that what the innermost function runs with can be read from the
    arguments alone:

    - a wrapper that takes only ``*args`` and ``**kwargs`` binds nothing itself;
    - a wrapper with a fixed signature (no ``*args`` or ``**kwargs``) that restates the
      function's own parameters, in order, each with no default or the function's own
      default, is handed every value and has no other to fall back on;
    - a wrapper that forwards and also names a parameter is transparent only for a
      parameter with no default on either side: a required one must be handed on, or the
      call fails. With a default on either side the wrapper may keep the value it is given
      while the function under it runs on its own."""
    named = [(name, parameter) for name, parameter in level.items() if not _variadic(parameter)]
    forwards = len(named) != len(level)
    if not forwards and [name for name, _ in named] != [
        name for name, parameter in inner.items() if not _variadic(parameter)
    ]:
        mismatch = next(
            (name for name, _ in named if name not in inner),
            next((name for name in inner if name not in level), ""),
        )
        return mismatch or (named[0][0] if named else "")
    empty = inspect.Parameter.empty
    for name, parameter in named:
        other = inner.get(name)
        if other is None or _variadic(other) or parameter.kind is not other.kind:
            return name
        if forwards:
            if parameter.default is not empty or other.default is not empty:
                return name
        elif parameter.default is not empty and not _same_value(parameter.default, other.default):
            return name
    return None


def _with_defaults(ctx: Any, tool_name: str, args: Any) -> tuple[Any, Optional[str]]:
    """The arguments the function will run with, or why they cannot be known.

    Python fills a default in when a function is called, after every hook, and a decorated
    tool is several functions: Pydantic AI builds the schema from the innermost one (it
    follows ``__wrapped__``) and calls the outermost. Its validation fills in the innermost
    function's defaults. A wrapper on the way that has a parameter of its own (a keyword
    default, a keyword it keeps from the function it wraps) is read by nothing, and whether
    its value reaches the function under it depends on its body, which cannot be read.

    So a wrapper must be transparent (``_opaque`` says what that is), or the call is
    refused. Every callable the call reaches is read (the kernel's walk over
    ``__wrapped__``, bound methods, callable objects and ``functools.partial``, whose own
    signature states what a call binds: its keywords are the defaults, its positionals are
    gone). Then the arguments are those of the innermost one: a default they leave out (a
    tool given its schema by hand, ``Tool.from_schema``, is not validated, so nothing has
    filled it in) is added, judged, and passed written out.

    Refused as well: a default that is not plain data, which cannot be shown to a policy; a
    callable that declares a signature of its own (``__signature__``), which hides the
    parameters Python binds; a stack that cannot be walked; and, for a tool that is not
    validated, a function that takes ``*args`` or ``**kwargs`` (it can fill a value in in
    its own body, as Pydantic AI's LangChain bridge does)."""
    function, takes_ctx, validated = _function_of(ctx, tool_name)
    if function is None or not isinstance(args, dict):
        return args, None
    unread = (
        "the function this tool runs, or a decorator on it, cannot be inspected, so the "
        "values it would run with cannot be read; call it from a plain function and "
        "register that one"
    )
    try:
        levels = _reached_parameters(function)
    except _DeclaredSignature:
        return args, (
            "a function this tool runs declares a signature of its own (`__signature__`), so "
            "the parameters Python binds, and their defaults, cannot be read"
        )
    except Exception:  # noqa: BLE001, a callable the kernel does not walk
        if validated and (isinstance(function, type) or inspect.isbuiltin(function)):
            # a class or a builtin wraps nothing, and Pydantic AI validated its own signature
            return args, None
        return args, unread
    inner = levels[-1]
    for level in levels[:-1]:
        own = _opaque(level, inner)
        if own is not None:
            return args, (
                f"a wrapper around the function this tool runs takes `{own}` in a way of its "
                "own: whether the function under it receives that value, or runs on a default "
                "of its own, depends on the wrapper's body, which cannot be read. A wrapper "
                "must take only `*args` and `**kwargs`, or restate the function's own "
                "parameters"
            )
    if not validated:
        for level in levels:
            for name, parameter in level.items():
                if _variadic(parameter):
                    return args, (
                        "this tool was given its schema by hand, so its arguments are not "
                        f"validated, and its function takes `{name}` as a variadic parameter: "
                        "a value it fills in itself cannot be read or judged; give the "
                        "function named parameters, or build the tool from its signature"
                    )
    missing: dict[str, Any] = {}
    for position, (name, parameter) in enumerate(inner.items()):
        if position == 0 and takes_ctx:
            continue
        if parameter.default is parameter.empty or name in args:
            continue
        if parameter.kind is parameter.POSITIONAL_ONLY:
            return args, (
                f"parameter `{name}` of the function this tool runs is positional-only and "
                "was left out, so its default cannot be written out for the policy to judge"
            )
        if not _json_data(parameter.default):
            return args, (
                f"parameter `{name}` of the function this tool runs has a default that is "
                "not plain data, so the value the function would run with cannot be judged; "
                "give it a JSON default (or None) and resolve the object inside the function"
            )
        missing[name] = parameter.default
    return ({**args, **missing} if missing else args), None


#: Settings that switch a provider-run tool on without it being among the native tools.
_PROVIDER_TOOL_SETTINGS = ("openai_native_tools",)
_PROVIDER_TOOL_BODY_KEYS = (
    "tools",
    "plugins",
    "web_search_options",
    "mcp_servers",
    "search_parameters",
)


def _provider_run_tools(request_context: Any) -> list[str]:
    """What this request would have the provider run, by name; empty if nothing."""
    found: set[str] = set()
    _collect_provider_run(
        request_context.model,
        request_context.model_settings,
        request_context.model_request_parameters,
        found,
        0,
    )
    return sorted(found)


def _collect_provider_run(
    model: Any, settings: Any, params: Any, found: set[str], depth: int
) -> None:
    candidates = getattr(model, "models", None)
    if isinstance(candidates, (list, tuple)) and candidates and depth < 8:
        # A fallback model resolves nothing itself: each model it may hand the request to
        # drops or swaps native tools, and merges in its own settings, when its turn comes.
        for candidate in candidates:
            _collect_provider_run(candidate, settings, params, found, depth + 1)
        return
    try:
        # The model's own resolution: it drops a native tool it does not support, or takes
        # the tool's local version instead, and merges in its own settings.
        settings, params = model.prepare_request(settings, params)
    except Exception:  # noqa: BLE001, unresolved: judge the request as it was handed over
        pass
    found.update(
        type(tool).__name__ for tool in params.native_tools if not isinstance(tool, ToolSearchTool)
    )
    if isinstance(settings, Mapping):
        found.update(f"`{key}`" for key in _PROVIDER_TOOL_SETTINGS if settings.get(key))
        body = settings.get("extra_body")
        if isinstance(body, Mapping):
            found.update(f"`extra_body.{key}`" for key in _PROVIDER_TOOL_BODY_KEYS if body.get(key))


#: How far into a toolset's attributes ``_held_toolsets`` looks, and how much it reads.
_HELD_DEPTH = 4
_HELD_LIMIT = 4000


def _attributes(obj: Any) -> list[Any]:
    """The values an object holds: its ``__dict__`` and its ``__slots__``."""
    values: list[Any] = []
    try:
        values.extend(vars(obj).values())
    except TypeError:
        pass
    for cls in type(obj).__mro__:
        slots = vars(cls).get("__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if name not in ("__dict__", "__weakref__"):
                try:
                    values.append(getattr(obj, name))
                except AttributeError:
                    pass
    return values


def _held_toolsets(toolset: Any) -> tuple[list[Any], bool]:
    """The toolsets this one holds, and whether the search was complete: in its attributes,
    in containers there (lists, tuples, sets, dicts, nested), and in plain objects there,
    ``_HELD_DEPTH`` levels down, nearest first. The search stops at each toolset it finds
    (that one is searched in its own turn) and gives up after ``_HELD_LIMIT`` values. A
    toolset reached only through a function is not found."""
    held: list[Any] = []
    seen: set[int] = {id(toolset)}
    pending: deque[tuple[Any, int]] = deque((value, 1) for value in _attributes(toolset))
    read = 0
    while pending and read < _HELD_LIMIT:
        value, depth = pending.popleft()
        read += 1
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, AbstractToolset):
            held.append(value)
        elif isinstance(value, (str, bytes, int, float, bool, type(None), type)) or callable(value):
            continue
        elif depth >= _HELD_DEPTH:
            continue
        elif isinstance(value, Mapping):
            pending.extend((item, depth + 1) for pair in value.items() for item in pair)
        elif isinstance(value, (list, tuple, set, frozenset)):
            pending.extend((item, depth + 1) for item in value)
        elif not _stock(type(value)):
            pending.extend((item, depth + 1) for item in _attributes(value))
    return held, not pending


def _stock(obj: Any) -> bool:
    module = getattr(obj, "__module__", None) or ""
    return module == "pydantic_ai" or module.startswith("pydantic_ai.")


def _rewriting_wrapper(
    toolset: Any, depth: int = 0, seen: Optional[set[int]] = None
) -> Optional[str]:
    """Why a toolset in this tree can change a call's arguments after the gate, or None.

    A toolset whose ``call_tool`` is Pydantic AI's own is trusted to pass a call on
    unchanged, and the toolsets it holds are checked in turn. One whose ``call_tool`` is
    the application's is refused when it holds other toolsets (it runs their calls, and a
    pass-through container would also hide what it holds), and when it replaces the
    ``call_tool`` of a Pydantic AI toolset it extends. An application toolset that holds
    none and extends none is the tool itself: its body is the application's, like a tool
    function's."""
    seen = set() if seen is None else seen
    if id(toolset) in seen:
        return None
    seen.add(id(toolset))
    if depth > 32:
        return "the toolsets are nested too deep to check"
    cls = type(toolset)
    if getattr(toolset, "process_tool_call", None) is not None:
        return (
            f"toolset {cls.__name__} has a `process_tool_call` function, which runs "
            "each call itself and could change the arguments after the gate judged them"
        )
    held, searched = _held_toolsets(toolset)
    try:
        own_instance = "call_tool" in vars(toolset)
    except TypeError:
        own_instance = False
    if own_instance or not _stock(getattr(cls, "call_tool", None)):
        if held:
            return (
                f"toolset {cls.__name__} wraps other tools and runs their calls itself "
                "(`call_tool`), so it could change the arguments after the gate judged them"
            )
        extends_stock = any(
            _stock(base)
            and "call_tool" in vars(base)
            and not getattr(vars(base)["call_tool"], "__isabstractmethod__", False)
            for base in cls.__mro__
        )
        if extends_stock:
            return (
                f"toolset {cls.__name__} replaces the `call_tool` of the Pydantic AI toolset "
                "it extends, so it could change the arguments after the gate judged them"
            )
        if not searched:
            return (
                f"toolset {cls.__name__} has its own `call_tool` and holds more than the gate "
                f"searches for other toolsets ({_HELD_LIMIT} values), so it cannot be told "
                "from one that runs other tools' calls"
            )
        return None
    for inner in held:
        reason = _rewriting_wrapper(inner, depth + 1, seen)
        if reason is not None:
            return reason
    return None


def _leaves(capability: AbstractCapability[Any]) -> list[AbstractCapability[Any]]:
    """The capabilities listed in this tree that are in effect from the start of a run:
    the members of a deferred group are loaded on demand, and are left out."""
    if isinstance(capability, CombinedCapability):
        if is_deferred_group(capability):
            return []
        return [leaf for child in capability.capabilities for leaf in _leaves(child)]
    return [capability]


def is_gated(agent: Agent[Any, Any]) -> bool:
    """True if a ``RecusalGate`` is among the capabilities the agent runs with now.

    Inside ``agent.override(spec=...)`` whose spec names capabilities, those replace the
    agent's own, and a gate among the originals no longer runs; this reads the ones in
    effect. A gate inside a group that is loaded on demand does not count: it judges
    nothing until the group is loaded. Nor does one a capability function returns or a
    ``WrapperCapability`` wraps: this reads the capabilities as listed, so for those it
    says False although the gate runs. Reads only this agent: an agent it calls from
    inside a tool function runs under its own capabilities and must be checked, and gated,
    itself.
    """
    override = getattr(agent, "_override_root_capability", None)
    current = override.get() if override is not None else None
    root = current.value if current is not None else agent.root_capability
    return any(isinstance(leaf, RecusalGate) for leaf in _leaves(root))
