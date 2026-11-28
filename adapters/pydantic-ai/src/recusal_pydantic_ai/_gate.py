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

import functools
import inspect
import types
import weakref
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

# The refusals the kernel's walk makes by a class of its own, each given a reason of its
# own here (``_walk_reason``). ``recusal`` 0.17.0, the oldest this package allows, has all
# of them: ``OpenDispatch``, ``ProxiedCallable`` and ``WeakProxy`` are new in it.
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
from recusal.invocation import json_data as _json_data
from recusal.invocation import own_parameters as _own_parameters
from recusal.invocation import reached_callables as _reached_callables

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.capabilities.hooks import BeforeToolExecuteHookFunc
    from pydantic_ai.models import ModelRequestContext

#: A Recusal policy: ``policy(tool_name, tool_input)`` returns a list of findings.
Policy = Callable[[str, dict[str, Any]], list[Finding]]

#: The kernel's walk refuses each of these by a class of its own; ``_walk_reason`` says
#: what each found. ``BoundArgumentKept`` and ``DeclaredSignature`` are handled apart.
_NAMED_REFUSALS = (
    _HiddenWrapper,
    _DispatchedAtCallTime,
    _OpenDispatch,
    _PartialMethodSignature,
    _ProxiedCallable,
)

#: The remedy a reason gives where the only one is to register another function. Taken
#: literally it moves what was refused into a body the gate does not read, so it says that,
#: and what the function must declare for its call to be judged.
_PLAIN = (
    "register in its place a plain function that declares every parameter the call can "
    "run with, defaults included, and passes each on by name; the gate judges that "
    "function's parameters and does not read its body or what it calls"
)

#: The remedy for a single-dispatch function, registered or not.
_BRANCH = (
    "register one plain function as the tool, declaring every parameter any implementation "
    "takes, and branch on the type inside its body; the gate judges that function's "
    "parameters and does not read its body, so a single-dispatch function called from it "
    "is not read"
)

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
        name (its ``__name__``). The name is all the policy is given, so a run with two
        output functions of one name is refused (``_shared_output_name``). Plain structured
        output has no output function and is not judged. A refusal asks the model again
        (``ModelRetry``); the function does not run."""
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
        shared = _shared_output_name(ctx, name, call is not None)
        if shared is not None:
            raise ModelRetry(_unjudgeable(name, shared, output, record))
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


def _output_functions(ctx: Any) -> Optional[list[Any]]:
    """The functions the run's output tools run, each once; None where the run context has
    no tool manager to read them from. An output tool's toolset holds one processor per
    tool, and a processor that runs a function holds that function's schema."""
    tools = getattr(getattr(ctx, "tool_manager", None), "tools", None)
    if not isinstance(tools, Mapping):
        return None
    found: list[Any] = []
    seen: set[int] = set()
    for tool in tools.values():
        for _ in range(32):
            processors = getattr(getattr(tool, "toolset", None), "processors", None)
            if isinstance(processors, Mapping) and id(processors) not in seen:
                seen.add(id(processors))
                for processor in processors.values():
                    schema = getattr(processor, "_function_schema", None)
                    function = getattr(schema, "function", None)
                    if function is not None and not any(
                        _same_callable(function, other) for other in found
                    ):
                        found.append(function)
            tool = getattr(tool, "source_tool", None)
            if tool is None:
                break
    return found


def _same_callable(left: Any, right: Any) -> bool:
    """True for one function, or one method of one object, listed twice."""
    return getattr(left, "__func__", left) is getattr(right, "__func__", right) and getattr(
        left, "__self__", None
    ) is getattr(right, "__self__", None)


def _shared_output_name(ctx: Any, name: str, by_tool: bool) -> Optional[str]:
    """Why the output function about to run cannot be told from another by its name, or
    None. Pydantic AI gives each output function its own output tool, and the model picks
    the tool; the policy is given the function's name. ``by_tool`` says the output arrived
    as a call to an output tool, so its own function is among the ones read: where it is
    not, the output functions could not be read, and that is refused too. A text output
    function is refused when an output tool runs a function of its name, so the name
    stands for one function."""
    functions = _output_functions(ctx)
    if functions is None:
        return None
    same = sum(getattr(function, "__name__", None) == name for function in functions)
    if same > (1 if by_tool else 0):
        return (
            f"more than one output function of this run is named `{name}`, and the policy "
            "is given only that name, so it could not tell which of them it is judging; "
            "give each output function a name of its own"
        )
    if by_tool and not same:
        return (
            "the output functions of this run cannot be read, so it cannot be told whether "
            f"another one is named `{name}` as well"
        )
    return None


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


def _positional(parameter: Any) -> bool:
    return parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)


def _same_value(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        return type(left) is type(right) and bool(left == right)
    except Exception:  # noqa: BLE001, a comparison that raises is not a match
        return False


def _as_bound(levels: list[Any], takes_ctx: bool) -> list[list[Any]]:
    """Each level's parameters, with the run context of a context-taking tool under the
    innermost function's name for it.

    Pydantic AI passes the context positionally and first, so a wrapper's first positional
    parameter is the context whatever the wrapper calls it and whether it is
    positional-only or not, unless it carries the name of another parameter of the
    function: that one a wrapper above can hand it by keyword, so it is read as the
    parameter it names. Nor is it renamed where the wrapper has another parameter under
    the context's name: then the first one is a parameter of the wrapper's own. The
    context is then one more leading positional-or-keyword parameter, held to the same
    rules as any other."""
    rows = [list(level.values()) for level in levels]
    inner = rows[-1]
    if not (takes_ctx and inner and _positional(inner[0])):
        return rows
    keyword = inspect.Parameter.POSITIONAL_OR_KEYWORD
    others = {parameter.name for parameter in inner[1:]}
    for row in rows:
        if row and _positional(row[0]) and row[0].name not in others:
            if all(parameter.name != inner[0].name for parameter in row[1:]):
                row[0] = row[0].replace(name=inner[0].name, kind=keyword)
    return rows


def _shape(parameters: list[Any]) -> list[tuple[str, Any]]:
    return [(parameter.name, parameter.kind) for parameter in parameters]


def _kind_differs(parameter: Any, other: Any) -> str:
    return (
        f"takes `{parameter.name}` as a {parameter.kind.description} parameter where the "
        f"function takes it as a {other.kind.description} one, so the two can bind the same "
        "call differently"
    )


def _opaque(
    level: list[Any], inner: list[Any], most: int, own: Optional[str] = None
) -> Optional[str]:
    """What keeps a wrapper from being transparent, in the words that follow "a wrapper
    around the function this tool runs", or None if it is transparent.

    ``level`` is the wrapper's parameters and ``inner`` the innermost function's, as
    ``_as_bound`` gives them; ``most`` is the largest number of positional parameters any
    wrapper above this one names; ``own`` is the wrapper's own name for its first
    parameter where ``_as_bound`` read it as the run context, and is the name a reason
    uses for it. Transparent means that what the innermost function runs with can be read
    from the arguments alone:

    - a wrapper that takes only ``*args`` and ``**kwargs`` binds nothing itself;
    - a wrapper that restates the function's own named parameters (same names, kinds and
      order, each with no default or the function's own default) is handed every value and
      has no other to fall back on. It has a fixed signature, or it also restates the
      function's ``**extra`` under the same name, and then only over a function that takes
      no ``*args`` and no positional-only parameter;
    - a wrapper that forwards (``**kwargs``) and also names a parameter is transparent only
      when binding by name and by position cannot differ, and when a parameter it does not
      hand on fails the call: the function takes no ``*args`` and no positional-only
      parameter (so Pydantic AI passes every value by keyword), each named parameter has
      the function's kind and no default on either side, and the positional-or-keyword
      ones are the function's leading ones in the function's order. One that forwards
      ``*args`` as well must name as many positional parameters as any wrapper above it:
      one above that names more and passes them on by position fills that ``*args``, and
      those values move up a place if this one keeps its own. A wrapper under it fills
      nothing: no value reaches this one's ``*args`` from below.
    """
    named = [parameter for parameter in level if not _variadic(parameter)]
    if not named:
        return None

    def called(parameter: Any) -> str:
        return own if own is not None and parameter is level[0] else str(parameter.name)

    theirs = [parameter for parameter in inner if not _variadic(parameter)]
    function = {parameter.name: parameter for parameter in theirs}
    for parameter in named:
        if parameter.name not in function:
            return (
                f"takes `{parameter.name}`, which is not a named parameter of the function: "
                "nothing reads the value it takes, and what the function then runs with "
                "depends on the wrapper's body, which cannot be read"
            )
    empty = inspect.Parameter.empty
    forwards = len(named) != len(level)
    by_position = any(
        parameter.kind in (parameter.VAR_POSITIONAL, parameter.POSITIONAL_ONLY)
        for parameter in inner
    )
    if forwards:
        restates = _shape(level) == _shape(inner) and not by_position
    else:
        restates = _shape(named) == _shape(theirs)
    if restates:
        for parameter in named:
            if parameter.default is not empty and not _same_value(
                parameter.default, function[parameter.name].default
            ):
                return (
                    f"gives `{called(parameter)}` a default that is not the function's own, "
                    "so the wrapper has a value to fall back on that nothing reads"
                )
        return None
    if not forwards:
        taken = {parameter.name for parameter in named}
        for parameter in theirs:
            if parameter.name not in taken:
                return (
                    "has a fixed signature that leaves out the function's parameter "
                    f"`{parameter.name}`, so the function cannot be handed the value the "
                    "policy judged for it"
                )
        for parameter, other in zip(named, theirs):
            if parameter.name != other.name:
                return (
                    "has a fixed signature that takes the function's parameters in another "
                    f"order (`{called(parameter)}` where the function has `{other.name}`), so "
                    "a value passed on by position reaches another parameter"
                )
        differs = next((p for p in named if p.kind is not function[p.name].kind), named[0])
        return _kind_differs(differs, function[differs.name])
    variadic = next((p for p in level if p.kind is p.VAR_POSITIONAL), None)
    if by_position:
        first = called(named[0])
        return (
            f"names `{first}` and also forwards, over a function that takes `*args` "
            "or a positional-only parameter: Pydantic AI passes such a function's values by "
            f"position, so `{first}` can be handed another parameter's value"
        )
    for parameter in named:
        other = function[parameter.name]
        if parameter.kind is not other.kind:
            return _kind_differs(parameter, other)
        if parameter.default is not empty or other.default is not empty:
            first = called(parameter)
            return (
                f"names `{first}` and also forwards, and `{first}` has a "
                "default (the wrapper's or the function's): the wrapper may keep the value "
                "it is given while the function under it runs on a default of its own"
            )
    positional = [p for p in named if p.kind is p.POSITIONAL_OR_KEYWORD]
    ours = [p.name for p in positional]
    if variadic is not None and 0 < len(ours) < most:
        first = called(positional[0])
        return (
            f"names `{first}` and also forwards positional arguments (`*{variadic.name}`), "
            "and another wrapper names more positional parameters above it: what that one "
            f"passes on by position moves into the place of `{first}` if this one does not "
            "hand it on, and the function runs with a value under another name than the "
            "policy read it under"
        )
    leading = [p.name for p in theirs if p.kind is p.POSITIONAL_OR_KEYWORD]
    for name, other_name in zip(ours, leading):
        if name != other_name:
            return (
                f"names `{name}` and also forwards, and `{name}` is not where the function "
                f"has it (the function has `{other_name}` there): handed on by position it "
                "reaches another parameter"
            )
    return None


def _opaque_wrapper(levels: list[Any], takes_ctx: bool) -> Optional[str]:
    """Why a wrapper among these levels (the outermost first, the function last) is not
    transparent, as the reason a call is refused with, or None. Each wrapper is held
    against the innermost function and against the wrappers above it."""
    *wrappers, function_row = _as_bound(levels, takes_ctx)
    above = 0
    for written, level in zip(levels, wrappers):
        own = next((str(parameter.name) for parameter in written.values()), None)
        if not level or own == level[0].name:
            own = None
        cause = _opaque(level, function_row, above, own)
        if cause is not None:
            return (
                f"a wrapper around the function this tool runs {cause}. A wrapper must take "
                "only `*args` and `**kwargs`, or restate the function's own parameters"
            )
        above = max(above, sum(map(_positional, level)))
    return None


#: How many callables ``_steps`` reads before it gives up. The kernel's walk stops at 32
#: levels, and a level is at most a method, its function, a partial and what they wrap.
_STEP_LIMIT = 256


def _steps(function: Any) -> list[Any]:
    """Every callable a call to ``function`` passes through, by each link the kernel's walk
    follows: a method's function, a partial's function, the ``__wrapped__`` a partial
    carries in its own dictionary, the ``__call__`` of a subclass of
    ``functools.partial``, ``__wrapped__``, what a ``functools.lru_cache`` wrapper calls
    where that is not what it names (``cache_target``), and a callable object's
    ``__call__`` bound to the object. Raises ``ValueError`` for more than it will read,
    and what ``cache_target`` raises.

    The rules this gate asks of each step (a single-dispatch function, a partial that
    binds by position, a bound method past a naming wrapper, a keyword partial) are thus
    asked of every callable the kernel's walk lays a level for (``_view``), and of the
    function of each method and partial it binds."""
    found: list[Any] = []
    pending = [function]
    while pending:
        if len(found) >= _STEP_LIMIT:
            raise ValueError("too many callables to read")
        step = pending.pop()
        found.append(step)
        if inspect.ismethod(step):
            pending.append(step.__func__)
        elif isinstance(step, functools.partial):
            pending.append(step.func)
            own = getattr(step, "__dict__", None)
            carried = own.get("__wrapped__") if type(own) is dict else None
            if carried is not None:
                pending.append(carried)
            call = inspect.getattr_static(type(step), "__call__", None)
            if inspect.isfunction(call):
                pending.append(types.MethodType(call, step))
        else:
            wrapped = getattr(step, "__wrapped__", None)
            if wrapped is not None:
                pending.append(wrapped)
            if type(step) is _CACHED:
                called = _cache_target(step)
                if called is not wrapped:
                    pending.append(called)
            if not (inspect.isfunction(step) or inspect.isbuiltin(step) or isinstance(step, type)):
                call = inspect.getattr_static(type(step), "__call__", None)
                if inspect.isfunction(call):
                    pending.append(types.MethodType(call, step))
    return found


def _base_alone(registry: Any, wrapper: Any) -> bool:
    """True if a single-dispatch registry holds the base implementation and nothing else:
    one entry, for ``object``, that is a function the walk reaches from ``wrapper``
    through ``__wrapped__``. ``register(object)`` puts another function in that entry, and
    then the base is not what runs. (``functools.wraps`` copies the registry to a wrapper
    above the single-dispatch function, so the base can be more than one step in.)"""
    if len(registry) != 1:
        return False
    only = registry.get(object)
    for _ in range(_STEP_LIMIT):
        wrapper = inspect.getattr_static(wrapper, "__wrapped__", None)
        if wrapper is None:
            return False
        if wrapper is only:
            return True
    return False


def _dispatches(step: Any) -> bool:
    """True for a single-dispatch function (``functools.singledispatch``: its ``registry``
    is a ``mappingproxy`` and its ``dispatch`` a callable, both read without running any
    code of the application's) or method (``functools.singledispatchmethod``, found behind
    the ``register`` its bound form carries) that can run anything but the base
    implementation the walk reaches: one registered beside it, or in its place. A
    callable that only has attributes of those names is not one."""
    registry = inspect.getattr_static(step, "registry", None)
    if type(registry) is types.MappingProxyType and callable(
        inspect.getattr_static(step, "dispatch", None)
    ):
        return not _base_alone(registry, step)
    owner = getattr(getattr(step, "register", None), "__self__", None)
    if isinstance(owner, functools.singledispatchmethod):
        return not _base_alone(owner.dispatcher.registry, owner.dispatcher)
    return False


#: The type of a ``functools.lru_cache`` or ``functools.cache`` wrapper. It has no
#: parameters of its own and no body that could keep a value, and the kernel's walk reads
#: it as the function it wraps, so that function's parameters show up twice.
_CACHED = type(functools.lru_cache(maxsize=None)(len))


def _without_its_own_wrapped(function: Any) -> Any:
    """A tool that is a ``functools.partial`` carrying a ``__wrapped__`` of its own (it was
    named with ``functools.update_wrapper`` or ``functools.wraps``), as the partial it is
    without that name; anything else as it is.

    A kernel later than 0.15.1 follows the name a partial carries, as one more level. This
    gate has rules of its own for such a partial (``_bound_over_a_wrapper``,
    ``_another_signature``: the signature Pydantic AI reads through that name is held
    against the function the call reaches), and they are written against the levels of
    the callables the call passes through. So the kernel is handed the same partial
    without the name, and those rules judge the name. A partial that also declares a
    ``__signature__`` is handed over as it is: the kernel refuses that first."""
    if not isinstance(function, functools.partial):
        return function
    own = getattr(function, "__dict__", None)
    if type(own) is not dict or "__wrapped__" not in own or "__signature__" in own:
        return function
    return functools.partial(function.func, *function.args, **function.keywords)


def _a_class_or_builtin(function: Any) -> bool:
    """True for a class or a builtin function, told by its own type: a ``weakref.proxy``
    of one answers ``isinstance`` for it and is neither."""
    return issubclass(type(function), type) or type(function) is types.BuiltinFunctionType


def _view(function: Any, validated: bool) -> tuple[list[Any], list[Any]]:
    """The parameters of every callable a call to ``function`` reaches, and beside each
    level the callable it is the level of. The callables are the kernel's walk,
    ``reached_callables``, and each level is ``own_parameters`` of the callable beside it,
    which is what ``reached_parameters`` returns for it by the kernel's own contract: one
    walk, and no walk of this gate's counted against the kernel's levels. Raises what the
    kernel's walk raises.

    (Before 0.2.1 this gate laid the callables of a walk of its own beside the kernel's
    levels and refused a stack where the two counts differed, which refused, beside
    ``recusal`` 0.16.0, a stack the kernel reads to the end:
    ``functools.update_wrapper(functools.partial(f), f)`` under another wrapper.)

    A class or a builtin wraps nothing and the kernel does not walk it: for a validated
    tool it is one level, the signature Pydantic AI validates against. One that names a
    ``__wrapped__`` is not one level (``inspect.signature`` reads past it), and what the
    kernel's walk raised for it is raised."""
    walked = _without_its_own_wrapped(function)
    try:
        callables = _reached_callables(walked)
        levels = [_own_parameters(callable_) for callable_ in callables]
    except _DeclaredSignature:
        raise
    except Exception:
        if not (validated and _a_class_or_builtin(function)):
            raise
        if getattr(function, "__wrapped__", None) is not None:
            raise
        return [inspect.signature(function).parameters], [function]
    return levels, callables


def _cached(function: Any) -> bool:
    """True for the level of a cache wrapper that is a copy of the level under it, bound to
    an object or not: a ``functools.lru_cache`` wrapper that calls the function it names.
    A cache named again for another function (``functools.wraps(f)`` above the cache)
    calls a wrapper ``f`` is not, and its level is that wrapper's own parameters, so it is
    not such a level; nor is one whose function cannot be told."""
    while inspect.ismethod(function):
        function = function.__func__
    if not isinstance(function, _CACHED):
        return False
    try:
        return _cache_target(function) is getattr(function, "__wrapped__", None)
    except Exception:  # noqa: BLE001, a cache that cannot be told counts as a wrapper
        return False


def _written(levels: list[Any], callables: list[Any]) -> list[Any]:
    """The levels of the callables that have parameters of their own: the level the walk
    shows for a cache wrapper that calls the function it names, which is a copy of the
    one under it, is left out (``_cached``). It is told by the callable beside it and by
    what the cache calls, and never by what it holds."""
    return [level for level, function in zip(levels, callables) if not _cached(function)]


def _calls_through_an_object(callables: list[Any], levels: list[Any]) -> Optional[str]:
    """Why the walk ends in a callable object's own ``__call__`` that takes ``*args`` or
    ``**kwargs``, or None.

    The kernel's walk passes through a callable object that names nothing as
    ``__wrapped__`` and reads its ``__call__``, which is then the last level: what that
    ``__call__`` hands the call to (a function the object keeps in an attribute, as a
    class-based decorator does) is not read. With named parameters only, the arguments are
    those parameters and the gate completes them. With ``*args`` or ``**kwargs`` the
    function the object keeps can run with defaults of its own that nothing showed: a
    class-based decorator that keeps the function and answers every other attribute from
    it (``__getattr__``), over a ``functools.partial``, on a tool that takes the run
    context, was judged ``{'table': 'prod'}`` while the function ran ``dry_run=True``
    (Pydantic AI takes ``*args`` for the context and puts the model's keys in
    ``**kwargs``)."""
    last = callables[-1]
    if not inspect.ismethod(last) or isinstance(last.__self__, type):
        return None
    if last.__func__ is not inspect.getattr_static(type(last.__self__), "__call__", None):
        return None
    variadic = [
        ("`*" if parameter.kind is parameter.VAR_POSITIONAL else "`**") + f"{name}`"
        for name, parameter in levels[-1].items()
        if _variadic(parameter)
    ]
    if not variadic:
        return None
    return (
        "the function this tool runs is reached through a callable object whose own "
        f"`__call__` takes {' and '.join(variadic)} and that names nothing it wraps (no "
        "`__wrapped__`), so what it hands the call to, and the defaults that runs with, "
        "cannot be read; this is refused although such an object may run correctly; give the "
        "object a `__wrapped__` that names the function it calls "
        f"(`functools.update_wrapper(self, fn)` in its `__init__`), or {_PLAIN}"
    )


def _carried_cause(function: Any, validated: bool, unread: str) -> Optional[str]:
    """For a tool that is a ``functools.partial`` carrying a ``__wrapped__`` of its own
    (``_without_its_own_wrapped`` hands the kernel the partial without it): why what that
    name leads to cannot be read, or None. Pydantic AI reads the tool's signature through
    the name, and the kernel's walk, handed the partial as it is, would follow it. So it is
    walked too, and refused for what the walk refuses there and for a callable object at
    its end (``_calls_through_an_object``): a partial of a forwarding helper named for a
    class-based decorator that answers every attribute from a ``functools.partial`` ran
    that partial's ``limit=7`` with the policy handed ``{'table': 't'}``."""
    if function is _without_its_own_wrapped(function):
        return None
    carried = function.__dict__["__wrapped__"]
    try:
        levels, callables = _view(carried, validated)
    except _DeclaredSignature as refused:
        return _declared_reason(refused)
    except (*_NAMED_REFUSALS, _BoundArgumentKept) as refused:
        return _walk_reason(refused)
    except Exception:  # noqa: BLE001, what the name leads to cannot be walked
        return _descriptor_cause(carried) or unread
    return _calls_through_an_object(callables, levels)


def _binds(step: Any) -> int:
    """How many values this callable hands in by position: the positional arguments of a
    ``functools.partial``, the object of a bound method."""
    if isinstance(step, functools.partial):
        return len(step.args)
    return 1 if inspect.ismethod(step) else 0


def _bound_past_a_wrapper(step: Any, reach: int) -> Optional[str]:
    """What makes the object of a bound method unreadable, in the words that follow "past a
    wrapper that names a positional parameter: ", or None.

    A bound method hands its object in by position, and the kernel's walk reads it as
    bound to the innermost function's leading parameter. Under a wrapper that names a
    positional parameter, that holds only if the wrapper hands the value on. It is assumed
    where a wrapper that keeps the value fails the call:

    - the innermost parameter bound that way has no default;
    - the innermost function takes nothing else by position (no ``*args``, no further
      positional-only parameter), so Pydantic AI passes no value that can move into its
      place;
    - no value a wrapper passes on by position can move into its place and leave a
      default to fill the gap. ``reach`` is how many values can travel by position: the
      most positional parameters any wrapper of the stack names (the run context of a
      context-taking tool is one), and the values other callables bind. The values
      after the kept one each move up a place, so the last of them leaves its own
      parameter empty, and the call fails unless that parameter has a default. A cache
      wrapper keeps nothing, so this is asked only where a wrapper with parameters of
      its own (``_written``) names a positional one.

    This is a rule about what a wrapper's body could do, and it is conservative: it also
    refuses a method decorator that restates the method's parameters, which no body that
    passes the object on can misuse. A ``functools.partial`` that binds by position is not
    read this way at all (``_bound_over_a_wrapper``)."""
    if not inspect.ismethod(step):
        return None
    under, callables = _view(step.__func__, False)
    if not any(_positional(p) for level in under[:-1] for p in level.values()):
        return None
    inner = list(under[-1].values())
    for position, parameter in enumerate(inner):
        if position < 1:
            if not _positional(parameter) or parameter.default is not parameter.empty:
                return (
                    f"whether that value reaches the function's `{parameter.name}`, or the "
                    "function runs on another value there, depends on the wrapper's body, "
                    "which cannot be read"
                )
        elif parameter.kind in (parameter.VAR_POSITIONAL, parameter.POSITIONAL_ONLY):
            return (
                f"the function's `{parameter.name}` is taken by position as well, so if the "
                "wrapper keeps that value the next one moves into its place, and whether it "
                "does depends on the wrapper's body, which cannot be read"
            )
    keepers = _written(under, callables)[:-1]
    if not any(_positional(p) for level in keepers for p in level.values()):
        return None
    later = [parameter for parameter in inner[1:] if _positional(parameter)]
    reach = min(reach, len(later))
    if reach and later[reach - 1].default is not inspect.Parameter.empty:
        return (
            f"the function's `{later[reach - 1].name}` has a default, so if the wrapper "
            "keeps that value and the ones after it are passed on by position, each moves "
            "up a place and the call does not fail; whether that happens depends on the "
            "wrappers' bodies, which cannot be read"
        )
    return None


def _bound_over_a_wrapper(step: Any) -> Optional[str]:
    """Why a ``functools.partial`` that binds a value by position cannot be read, or None.

    The partial's own signature, which is what Pydantic AI and the walk read, drops the
    leading parameter of EVERY callable under it, with that parameter's default: a wrapper
    under the partial that names a parameter of its own then looks like one that names
    none, and which parameter the bound value fills, what a wrapper hands on under
    another name and what it falls back on are all out of sight. No reasoning about what
    a wrapper's body would do holds here, so the rule is about shape alone. Under a
    partial that binds by position, down to the function, there may only be:

    - a wrapper that takes nothing but ``*args`` and ``**kwargs``;
    - another ``functools.partial`` (its own level shows what is under it).

    (A cache wrapper under any partial never gets here: the kernel's walk cannot read a
    partial of one, and the call is refused as one that cannot be inspected.)

    Refused as well: such a partial that carries ``__wrapped__`` (it was named with
    ``functools.update_wrapper``, and ``inspect.signature`` then reads past it), and one
    that binds a positional-only parameter of a function that takes ``**kwargs`` (an
    argument sent under that parameter's name is not a duplicate to Python: it lands in
    ``**kwargs`` while the function runs the bound value)."""
    if not (isinstance(step, functools.partial) and step.args):
        return None
    advice = f"or {_PLAIN}"
    if inspect.getattr_static(step, "__wrapped__", None) is not None:
        return (
            "a `functools.partial` that binds a value by position was named with "
            "`functools.update_wrapper` or `functools.wraps`, which makes "
            "`inspect.signature`, and so Pydantic AI, read past the partial to the function "
            "under it: the value the partial binds is not in the signature the arguments "
            "are validated against; set `__name__` and `__qualname__` on the partial instead, "
            f"{advice}"
        )
    levels = _view(step.func, False)[0]
    for level in levels[:-1]:
        for parameter in level.values():
            if not _variadic(parameter):
                return (
                    "a `functools.partial` binds a value by position past a wrapper that "
                    f"names a parameter (`{parameter.name}`): which parameter the value "
                    "fills, and what each wrapper then keeps, cannot be read; bind it by "
                    f"keyword, {advice}"
                )
    inner = list(levels[-1].values())
    takes_keywords = any(parameter.kind is parameter.VAR_KEYWORD for parameter in inner)
    # positional-only parameters come first, so the first one says whether one is bound
    if inner and inner[0].kind is inspect.Parameter.POSITIONAL_ONLY and takes_keywords:
        return (
            "a `functools.partial` binds the positional-only parameter "
            f"`{inner[0].name}` of a function that also takes `**kwargs`: an argument "
            "sent under that name is not refused as a duplicate, it lands in `**kwargs` "
            "while the function runs the bound value; make the parameter an ordinary "
            f"one, {advice}"
        )
    return None


def _keyword_past_a_parameter(step: Any, levels: list[Any]) -> Optional[str]:
    """Why a keyword a ``functools.partial`` binds does not reach the parameter it names,
    or None. ``levels`` is every callable under the partial, read without its keywords.

    Python binds a keyword to a positional-or-keyword or keyword-only parameter. One that
    names a positional-only parameter is not bound to it: the call fails, or a ``**kwargs``
    takes the value and the parameter keeps its own default, and the partial's signature,
    which is what Pydantic AI and the walk read, no longer shows that parameter at all."""
    for level in levels:
        for key in step.keywords:
            parameter = level.get(key)
            if parameter is not None and parameter.kind not in (
                parameter.POSITIONAL_OR_KEYWORD,
                parameter.KEYWORD_ONLY,
            ):
                whose = (
                    "the function this tool runs"
                    if level is levels[-1]
                    else "a wrapper around the function this tool runs"
                )
                return (
                    f"a `functools.partial` binds `{key}` by keyword, and {whose} takes "
                    f"`{key}` as a {parameter.kind.description} parameter, which a keyword "
                    "does not reach: the value that parameter runs with cannot be read "
                    "through the partial; make that parameter positional-or-keyword or "
                    f"keyword-only, or {_PLAIN}"
                )
    return None


def _unreadable_step(
    function: Any, levels: list[Any], callables: list[Any], takes_ctx: bool, validated: bool
) -> Optional[str]:
    """Why a callable on the way to the code that runs cannot be read from signatures, or
    None. ``levels`` and ``callables`` are the walk of the whole stack (``_view``). Raises
    what ``_steps`` and the kernel's walk raise.

    A ``functools.partial`` with keywords is read twice. Its own signature states what a
    call binds, and it also turns every parameter from the first keyword-bound one on
    into a keyword-only one and drops ``*args``, on a wrapper's level and on the
    function's alike, so a wrapper that takes a parameter as another kind than the
    function does looks like one that restates it. So the levels under such a partial are
    read without it as well, and a stack that is refused without the partial is refused
    with it. (A partial that also binds by position has only pure forwarders under it,
    ``_bound_over_a_wrapper``, so leaving its positional values out too changes nothing.)"""
    steps = _steps(function)
    unbound = {
        index: _view(step.func, validated)
        for index, step in enumerate(steps)
        if isinstance(step, functools.partial) and step.keywords
    }
    named = max(
        (
            sum(map(_positional, level.values()))
            for view in ((levels, callables), *unbound.values())
            for level in _written(*view)[:-1]
        ),
        default=0,
    )
    bound = sum(map(_binds, steps))
    for index, step in enumerate(steps):
        if _dispatches(step):
            return _DISPATCHED
        cause = _bound_over_a_wrapper(step)
        if cause is not None:
            return cause
        cause = _bound_past_a_wrapper(step, max(named, int(takes_ctx)) + bound - _binds(step))
        if cause is not None:
            return (
                "a bound method hands its object in by position past a wrapper that names "
                f"a positional parameter: {cause}; give that wrapper only `*args` and "
                f"`**kwargs`, or {_PLAIN}"
            )
        if index in unbound:
            under = unbound[index][0]
            cause = _keyword_past_a_parameter(step, under)
            if cause is None:
                cause = _opaque_wrapper(under, takes_ctx)
            if cause is not None:
                return cause
    return None


_DISPATCHED = (
    "the function this tool runs is a single-dispatch function "
    "(`functools.singledispatch` or `singledispatchmethod`) with an "
    "implementation registered beside its base one or in its place: the "
    "implementation that runs is chosen at call time, by the type of an "
    f"argument, so its parameters and defaults cannot be read; {_BRANCH}"
)


_PARTIALMETHOD = (
    "the function this tool runs is a `functools.partialmethod` over a callable that does "
    "not bind as a method (a callable object, a class-based decorator), or a wrapper of "
    "one: the signature it shows is that of what the partialmethod binds, not its own, so "
    "what runs under it cannot be read; put the partialmethod over a plain function, "
    f"which is read in full, or {_PLAIN}"
)


def _bound_partial_cause(function: Any) -> Optional[str]:
    """This gate's own reason for a ``functools.partial`` that binds by position somewhere
    on the way (``_bound_over_a_wrapper``), for a call the kernel's walk has refused
    already; None where there is none or the callables cannot be read that far."""
    try:
        for step in _steps(function):
            cause = _bound_over_a_wrapper(step)
            if cause is not None:
                return cause
    except Exception:  # noqa: BLE001, the kernel's refusal stands as it is
        return None
    return None


_OPEN_DISPATCH = (
    "the function this tool runs is, or runs through, a single-dispatch function "
    "(`functools.singledispatch` or `singledispatchmethod`) with nothing registered but "
    "its base: an implementation can be registered on it at any moment, also between this "
    "check and the call (by another tool of the same turn, or by a wrapper that registers "
    "one on its first use), and would then run with parameters and defaults no policy "
    "read; this is refused although a single-dispatch function that nothing is ever "
    f"registered on runs correctly; {_BRANCH}"
)


_WEAK_PROXY = (
    "the function this tool runs is, or runs through, a `weakref.proxy`: a proxy shows the "
    "attributes and the class of the object it stands for and is not that object, so what "
    "a call to it runs cannot be told (a cache named again for another function, a "
    "single-dispatch function and a class-based decorator's own `__call__` are each told "
    "by the type of the callable itself); this is refused although a proxy of a plain "
    "function runs correctly; register the function itself, not a proxy of it"
)


_PROXIED = (
    "a callable on the way to the function this tool runs answers for another object, so "
    "what a call to it runs cannot be read"
)


_PROXIED_REMEDY = (
    "The ordinary case is a class-based decorator that keeps the function in an attribute "
    "and hands every other attribute on to it (`__getattr__`), over a function that names "
    "a `__wrapped__` of its own; it runs correctly and is refused all the same. Call "
    "`functools.update_wrapper(self, fn)` in its `__init__` and it is read to the end. "
    f"Otherwise {_PLAIN}"
)

#: Said after the kernel's own words, which end with the kernel's remedy; where that is to
#: register a plain function, this is what the gate does with one.
_KERNEL_REMEDY = (
    "a plain function registered in its place is judged by its own parameters, and its "
    "body, and what it calls, is not read: declare on it every parameter the call can run "
    "with, defaults included"
)

#: The remedy for a declared signature, by the attribute the kernel found.
_UNDECLARE = {
    "__signature__": "delete the `__signature__` so the parameters Python binds are read",
    "__text_signature__": (
        "delete the `__text_signature__` so the parameters Python binds are read"
    ),
    "__code__": (
        "give the object a `__wrapped__` that names the function it calls "
        "(`functools.update_wrapper(self, fn)` in its `__init__`)"
    ),
}


#: What each declared signature is, by the attribute the kernel's walk found
#: (``DeclaredSignature.declared``).
_DECLARED = {
    "__signature__": "declares a signature of its own (`__signature__`)",
    "__text_signature__": (
        "declares a text signature of its own (`__text_signature__`) that is not the interpreter's"
    ),
    "__code__": (
        "is an object that shows a `__code__`, which `inspect.signature` can read as a "
        "function's in place of the parameters of its `__call__`"
    ),
}


def _declared_reason(refused: Any) -> str:
    """The reason for a ``DeclaredSignature``, naming the attribute the kernel found. One
    raised without it (as code written for ``recusal`` 0.16.0 raises it) is a declared
    signature of no named kind."""
    declared = getattr(refused, "declared", None)
    what, undo = "declares a signature of its own", None
    if isinstance(declared, str):
        what, undo = _DECLARED.get(declared, what), _UNDECLARE.get(declared)
    remedy = f"{undo}, or {_PLAIN}" if undo is not None else _PLAIN
    return (
        f"a callable this tool runs {what}, so the parameters Python binds, and their "
        "defaults, cannot be read; this is refused also where the declaration matches "
        f"them; {remedy}"
    )


def _walk_reason(refused: BaseException) -> str:
    """The reason a call is refused with when the kernel's walk raises one of the
    refusals it names by a class of its own. A single-dispatch function with an
    implementation registered gets the reason this gate gives one it finds itself, and
    one with nothing registered (``OpenDispatch``) a reason of its own, which does not say
    that an implementation is registered. A weak proxy and a partialmethod get their
    cause and what to do. A callable that answers for another object
    (``ProxiedCallable``, three shapes) gets the kernel's own words, which say which shape
    and name no value, and the remedy for the ordinary one. A hidden wrapper, a bound
    argument and a cache get the kernel's own words, which name a parameter and never a
    value, and what to do."""
    if isinstance(refused, _DispatchedAtCallTime):
        return _DISPATCHED
    if isinstance(refused, _OpenDispatch):
        return _OPEN_DISPATCH
    if isinstance(refused, _PartialMethodSignature):
        return _PARTIALMETHOD
    if isinstance(refused, _WeakProxy):
        return _WEAK_PROXY
    if isinstance(refused, _ProxiedCallable):
        return f"{_PROXIED} (the kernel's walk reports: {refused}). {_PROXIED_REMEDY}"
    if isinstance(refused, _BoundArgumentKept):
        return f"the function this tool runs cannot be read: {refused}; {_KERNEL_REMEDY}"
    if isinstance(refused, _UnreadCache):
        return f"a cache on the function this tool runs cannot be read: {refused}; {_KERNEL_REMEDY}"
    return (
        "a decorator on the function this tool runs holds a second wrapper of that "
        "function (in its closure or on the decorator object), which is not reached "
        f"through `__wrapped__` and so is not read like one applied directly: {refused}; "
        f"{_PLAIN}"
    )


_DESCRIPTOR = (
    "a decorator on the function this tool runs is an object whose class defines `__get__` "
    "(a class-based decorator that binds as a method, such as `cachetools.cached`, or a "
    "`staticmethod` object called as a method): `inspect.signature` reads such an object as "
    "a built-in and shows no parameters for it, so what it runs cannot be read; give its "
    f"class no `__get__`, or {_PLAIN}"
)


def _descriptor_cause(function: Any) -> Optional[str]:
    """The reason for a stack the kernel's walk could not read because a callable on it is
    an object ``inspect`` takes for a method descriptor (its class defines ``__get__`` and
    no ``__set__``) called through a ``__call__`` written in Python, or a ``staticmethod``
    object: the test ``inspect`` itself makes, and the kernel after it. And the reason for
    a ``weakref.proxy`` told by its own type, where the walk failed before its end, where
    it raises ``WeakProxy``: on Python 3.11.1 ``inspect.signature`` supports no proxy of a
    callable object, and the walk raised a plain ``ValueError``. None where there is none
    or the callables cannot be read that far."""
    try:
        for step in _steps(function):
            if type(step) in weakref.ProxyTypes:
                return _WEAK_PROXY
            if inspect.isfunction(step) or inspect.ismethod(step) or isinstance(step, type):
                continue
            if not inspect.ismethoddescriptor(step):
                continue
            call = inspect.getattr_static(type(step), "__call__", None)
            if isinstance(step, staticmethod) or inspect.isfunction(call):
                return _DESCRIPTOR
    except Exception:  # noqa: BLE001, the refusal stands with the generic reason
        return None
    return None


def _filled_by_position(function: Any, args: dict[str, Any]) -> Optional[str]:
    """Why the arguments name a parameter that a ``functools.partial`` or the object of a
    bound method already fills by position, or None.

    Such a parameter is not in the signature Pydantic AI validates against, so a name the
    model sends for it is refused there, unless the function takes ``**kwargs``: then it
    is validated into them, judged, and the call fails with a second value for the
    parameter (``TypeError``), which left the run as an exception. It is refused here, as
    an answer the model reads."""
    try:
        for step in _steps(function):
            if isinstance(step, functools.partial) and step.args:
                inner, count = _view(step.func, False)[0][-1], len(step.args)
            elif inspect.ismethod(step):
                inner, count = _view(step.__func__, False)[0][-1], 1
            else:
                continue
            taken = [name for name, p in inner.items() if _positional(p)][:count]
            sent = sorted(set(taken) & set(args))
            if sent:
                return (
                    f"the arguments name `{sent[0]}`, a parameter of the function this tool "
                    "runs that a `functools.partial` or the object of a method already fills "
                    "by position, so the call cannot run with a value under that name as "
                    "well (Python refuses it as a second value for the parameter, or as a "
                    "positional-only parameter passed by keyword); leave "
                    f"`{sent[0]}` out of the arguments"
                )
    except Exception:  # noqa: BLE001, the walk has been read once already
        return None
    return None


def _another_signature(function: Any, inner: Any) -> Optional[str]:
    """Why the signature Pydantic AI reads for this tool is not the function the call
    reaches, or None. Pydantic AI validates the arguments against
    ``inspect.signature(function)`` and builds the call from it (which values go by
    position). That reading follows ``__wrapped__`` from the callable it is handed; the
    gate's walk follows what Python binds. For every tool the two must be the same
    parameters: the same names, kinds, order and defaults as the innermost level."""
    different = (
        "the signature Pydantic AI reads for this tool (`inspect.signature`, which follows "
        "`__wrapped__`) is not that of the function the call reaches, so the arguments are "
        "validated, and passed by position, for another function; a `functools.partial` "
        "named with `functools.update_wrapper` or `functools.wraps` does this: set "
        f"`__name__` and `__qualname__` on the partial instead, or {_PLAIN}"
    )
    try:
        read = list(inspect.signature(function).parameters.values())
    except Exception:  # noqa: BLE001, no signature to hold against the walk
        return different
    reached = list(inner.values())
    if len(read) != len(reached):
        return different
    for theirs, ours in zip(read, reached):
        if theirs.name != ours.name or theirs.kind is not ours.kind:
            return different
        if not _same_value(theirs.default, ours.default):
            return different
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
    refused. Every callable on the stack is read (the kernel's walk over
    ``__wrapped__``, bound methods, callable objects and ``functools.partial``, whose own
    signature states what a call binds: its keywords are the defaults, its positionals are
    gone; the levels under a partial with keywords are read without them as well, and
    ``_unreadable_step`` says why). Then the arguments are those of the innermost one: a
    default they leave out (a tool given its schema by hand, ``Tool.from_schema``, is not
    validated, and a hook of the application's can hand arguments in in place of
    validation, so nothing has filled it in) is added, judged, and passed written out. A
    class or a builtin, which the kernel does not walk, is read by the signature Pydantic
    AI validates against, for a validated tool only.

    Refused as well: a default that is not plain data, which cannot be shown to a policy; a
    callable that declares a signature of its own (a ``__signature__``, a
    ``__text_signature__`` or a ``__code__`` on an object, ``_declared_reason``), which
    hides the parameters Python binds; a stack that cannot be walked; a single-dispatch
    function that can run another implementation than its base one, picked at call time
    (found here, ``_dispatches``, or by the kernel's walk), and one with nothing
    registered yet, on which one can be registered before the call (``OpenDispatch``); a
    ``weakref.proxy`` anywhere on the way (``WeakProxy``) and a callable that answers for
    another object (``ProxiedCallable``); a decorator that holds a second wrapper the
    kernel's walk refuses (``_walk_reason``); a class-based decorator that is a
    descriptor (``_descriptor_cause``); a walk that ends in a callable object's own
    ``__call__`` with ``*args`` or ``**kwargs`` (``_calls_through_an_object``); a name the
    arguments give for a parameter a partial or a method's object fills by position
    (``_filled_by_position``); a
    positional argument bound past a naming wrapper where keeping it would not fail the
    call (``_bound_past_a_wrapper``); a keyword a partial binds that does not reach the
    parameter it names (``_keyword_past_a_parameter``); a value for the function's
    ``*args`` that is not a list (it reaches the hook only where a hook of the
    application's skipped validation, and Pydantic AI spreads it); and, for a tool that
    is not validated, a function or wrapper that takes ``*args`` or ``**kwargs`` (it can
    fill a value in in its own body, as Pydantic AI's LangChain bridge does)."""
    function, takes_ctx, validated = _function_of(ctx, tool_name)
    if function is None or not isinstance(args, dict):
        return args, None
    unread = (
        "the function this tool runs, or a decorator on it, cannot be inspected, so the "
        f"values it would run with cannot be read; {_PLAIN}"
    )
    try:
        # A class or a builtin wraps nothing, and Pydantic AI validates against the signature
        # it shows. A hook can hand arguments in in place of validation, so that signature
        # is read as well (`_view`), and a default they leave out is completed below.
        levels, callables = _view(function, validated)
    except _DeclaredSignature as refused:
        return args, _declared_reason(refused)
    except _BoundArgumentKept as refused:
        # this gate's own rule for a partial that binds by position says more about it
        return args, _bound_partial_cause(function) or _walk_reason(refused)
    except _NAMED_REFUSALS as refused:
        return args, _walk_reason(refused)
    except Exception:  # noqa: BLE001, a callable the kernel does not walk
        # a partial that binds by position and carries a name of its own, further in than
        # the tool itself, and a class-based decorator that is a descriptor, are ones this
        # gate can say more about
        return args, _bound_partial_cause(function) or _descriptor_cause(function) or unread
    try:
        unreadable = _unreadable_step(function, levels, callables, takes_ctx, validated)
    except _DeclaredSignature as refused:
        return args, _declared_reason(refused)
    except (*_NAMED_REFUSALS, _BoundArgumentKept) as refused:
        # the levels under a partial with keywords, walked without it
        return args, _walk_reason(refused)
    except Exception:  # noqa: BLE001, a step that cannot be read is not let through
        return args, unread
    if unreadable is not None:
        return args, unreadable
    through = _calls_through_an_object(callables, levels) or _carried_cause(
        function, validated, unread
    )
    if through is not None:
        return args, through
    inner = levels[-1]
    if validated:
        another = _another_signature(function, inner)
        if another is not None:
            return args, another
    opaque = _opaque_wrapper(levels, takes_ctx)
    if opaque is not None:
        return args, opaque
    filled = _filled_by_position(function, args)
    if filled is not None:
        return args, filled
    if not validated:
        for level in levels:
            for name, parameter in level.items():
                if _variadic(parameter):
                    whose, fix = (
                        ("its function", "give the function named parameters")
                        if level is inner
                        else (
                            "a wrapper around its function",
                            "restate the function's parameters in the wrapper",
                        )
                    )
                    return args, (
                        "this tool was given its schema by hand, so its arguments are not "
                        f"validated, and {whose} takes `{name}` as a variadic parameter: "
                        f"a value it fills in itself cannot be read or judged; {fix}, or "
                        "build the tool from its signature"
                    )
    missing: dict[str, Any] = {}
    for position, (name, parameter) in enumerate(inner.items()):
        if position == 0 and takes_ctx:
            continue
        if parameter.kind is parameter.VAR_POSITIONAL and type(args.get(name, [])) is not list:
            return args, (
                f"the value for `*{name}` of the function this tool runs is not a list: "
                "Pydantic AI hands the function its items one by one, so the function would "
                "not run with the value the policy judged"
            )
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
