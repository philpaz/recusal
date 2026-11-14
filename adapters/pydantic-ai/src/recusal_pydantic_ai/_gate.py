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
validated arguments; output tools (structured output) are not tool calls and skip it.

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
from typing import Any, Callable, List, Optional

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, CombinedCapability
from pydantic_ai.exceptions import ToolFailed
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

from recusal import Finding, decide
from recusal.invocation import detached as _detached

#: A Recusal policy: ``policy(tool_name, tool_input)`` returns a list of findings.
Policy = Callable[[str, dict], List[Any]]


def _unjudgeable(tool_name: str, detail: str, proposal: Any, record: dict) -> str:
    """The reason for a call the gate refuses before any policy runs.

    Without recording arguments it is the plain text. With them (the audited gate) the
    refusal goes through ``decide`` like every other, so it is one entry on the log, bound
    to the call as the model proposed it, and the reason the model reads is the one on the
    record.
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


def recusal_before_tool_execute(policy: Policy):
    """Return a ``before_tool_execute`` hook that judges each tool call before it runs.

    For ``Hooks(before_tool_execute=...)``. A refusal is raised as ``ToolFailed(reason)``,
    which Pydantic AI turns into a failed tool result the model reads on its next turn;
    the returned arguments run the tool unchanged.

    ``Hooks`` declares no ordering, so this form must be the last capability in the list:
    a hook that runs after it can change the arguments it judged.
    """

    async def gate(ctx: Any, *, call: ToolCallPart, tool_def: ToolDefinition, args: dict) -> dict:
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

    async def before_tool_execute(
        self, ctx: Any, *, call: ToolCallPart, tool_def: ToolDefinition, args: dict
    ) -> dict:
        reason = _refusal(call.tool_name, args, self.policy)
        if reason is not None:
            raise ToolFailed(reason)
        return args


def _leaves(capability: Any) -> list:
    if isinstance(capability, CombinedCapability):
        return [leaf for child in capability.capabilities for leaf in _leaves(child)]
    return [capability]


def is_gated(agent: Agent) -> bool:
    """True if a ``RecusalGate`` is among the agent's own capabilities.

    Reads only this agent: an agent it calls from inside a tool function runs under its own
    capabilities and must be checked, and gated, itself.
    """
    return any(isinstance(leaf, RecusalGate) for leaf in _leaves(agent.root_capability))
