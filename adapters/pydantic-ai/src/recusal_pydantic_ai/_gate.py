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
application to run (an ``ExternalToolset``) in ``handle_deferred_tool_calls``. A toolset
that wraps other tools and runs their calls itself runs after every hook, so a run with
one (other than those Pydantic AI ships) has every call refused (``get_wrapper_toolset``).
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, CombinedCapability
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.output import OutputContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset, WrapperToolset
from pydantic_ai.toolsets.set_metadata import SetMetadataToolset

from recusal import Finding, decide
from recusal.invocation import detached as _detached

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.capabilities.hooks import BeforeToolExecuteHookFunc

#: A Recusal policy: ``policy(tool_name, tool_input)`` returns a list of findings.
Policy = Callable[[str, dict[str, Any]], list[Finding]]


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

    Pydantic AI hands the hook the validated argument dict the tool will be called with, so
    the policy judges exactly what would run. A non-dict is refused: the tool would not
    accept it either, and the policy cannot read it.

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
    tool. Only another ``innermost`` capability listed after the gate still runs after it;
    list the gate last.

    ``policy`` is required. A gate has no default policy: what to refuse is the adopter's
    decision, never the adapter's.
    """

    policy: Policy

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
        if unseen:
            raise ToolFailed(_unjudgeable(call.tool_name, unseen, self._proposal(call), record))
        reason = _refusal(
            call.tool_name, args, self.policy, proposal=self._proposal(call), **record
        )
        if reason is not None:
            raise ToolFailed(reason)
        return args

    def get_wrapper_toolset(self, toolset: AbstractToolset[Any]) -> Optional[AbstractToolset[Any]]:
        """Read the run's assembled toolset. The execute hooks run before the toolset does,
        so a toolset that wraps another and runs its calls itself (``call_tool``) can change
        the arguments after the gate judged them. When one is in the tree, every tool of the
        run is marked, and ``before_tool_execute`` refuses each call, on the record. The
        wrappers Pydantic AI ships (prefixing, renaming, filtering, approval, tool search,
        durable execution) route or re-validate a call and do not change its arguments."""
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
        if not output_context.has_function:
            return output
        call = output_context.tool_call
        record = self._record(call.tool_call_id if call is not None else None, SURFACE_OUTPUT)
        name = output_context.function_name
        if name is None:
            raise ModelRetry(
                _unjudgeable(
                    "output", "the output function that would run is not known", output, record
                )
            )
        args = output if isinstance(output, dict) else {"output": output}
        reason = _refusal(name, args, self.policy, proposal=self._proposal(call), **record)
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
        ``ExternalToolset``, a tool that raises ``CallDeferred``) before they leave it: a
        refused call is answered with the refusal, which the model reads; an allowed one is
        handed on with exactly the arguments judged. Calls waiting for approval are judged
        when they run, by ``before_tool_execute``."""
        results = DeferredToolResults()
        for call in requests.calls:
            try:
                args: Any = call.args_as_dict()
            except Exception:  # noqa: BLE001, arguments that are not an object are refused
                args = call.args
            reason = _refusal(
                call.tool_name,
                args,
                self.policy,
                proposal=self._proposal(call),
                **self._record(call.tool_call_id, SURFACE_DEFERRED),
            )
            if reason is not None:
                results.calls[call.tool_call_id] = ToolFailed(reason)
        return results if results.calls else None


#: The audit surfaces, one per hook that adjudicates.
SURFACE_TOOL = "pydantic_ai.before_tool_execute"
SURFACE_OUTPUT = "pydantic_ai.before_output_process"
SURFACE_DEFERRED = "pydantic_ai.handle_deferred_tool_calls"

#: The tool metadata key ``get_wrapper_toolset`` sets on every tool of a run it cannot see
#: all of; its value is the reason.
_RUN_REFUSAL = "recusal_unjudgeable_run"


def _rewriting_wrapper(toolset: Any, depth: int = 0) -> Optional[str]:
    """Why a toolset in this tree can change a call's arguments after the gate, or None."""
    if depth > 32:
        return "the toolsets are nested too deep to check"
    if isinstance(toolset, WrapperToolset):
        cls = type(toolset)
        runs_calls = cls.call_tool is not WrapperToolset.call_tool or "call_tool" in vars(toolset)
        if runs_calls and not cls.__module__.startswith("pydantic_ai."):
            return (
                f"toolset {cls.__name__} wraps other tools and runs their calls itself "
                "(`call_tool`), so it could change the arguments after the gate judged them"
            )
        return _rewriting_wrapper(toolset.wrapped, depth + 1)
    if isinstance(toolset, CombinedToolset):
        for inner in toolset.toolsets:
            reason = _rewriting_wrapper(inner, depth + 1)
            if reason is not None:
                return reason
    return None


def _leaves(capability: AbstractCapability[Any]) -> list[AbstractCapability[Any]]:
    if isinstance(capability, CombinedCapability):
        return [leaf for child in capability.capabilities for leaf in _leaves(child)]
    return [capability]


def is_gated(agent: Agent[Any, Any]) -> bool:
    """True if a ``RecusalGate`` is among the capabilities the agent runs with now.

    Inside ``agent.override(spec=...)`` whose spec names capabilities, those replace the
    agent's own, and a gate among the originals no longer runs; this reads the ones in
    effect. Reads only this agent: an agent it calls from inside a tool function runs
    under its own capabilities and must be checked, and gated, itself.
    """
    override = getattr(agent, "_override_root_capability", None)
    current = override.get() if override is not None else None
    root = current.value if current is not None else agent.root_capability
    return any(isinstance(leaf, RecusalGate) for leaf in _leaves(root))
