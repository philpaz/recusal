"""
Claude adapter, put the gate in front of a Claude agent's tool calls.

Claude can now run long, autonomous, multi-tool agent loops. This adapter is the
deterministic policy seam: before a proposed tool call executes, you adjudicate
the evidence and the gate can **refuse**, and Claude adapts instead of acting.
"Builders cannot grade their own work" applied to an agent's own tool use.

Two integration surfaces, both from Anthropic's documented agent patterns:

1. Manual agentic loop (`client.messages.create` in a while-loop). Anthropic's
   own guidance: use the manual loop "when you need human-in-the-loop approval
   before each tool execution." Swap the human for a deterministic gate with
   ``gate_tool_use``, on FAIL it returns a ``tool_result`` block with
   ``is_error: true``, which Claude reads and self-corrects from.

2. Managed Agents with ``permission_policy: {"type": "always_ask"}``. The session
   idles on a tool call awaiting a ``user.tool_confirmation`` event;
   ``tool_confirmation`` is the deterministic decider behind that confirmation
   (``allow`` / ``deny`` with a reason fed back to the agent).

This module has **no dependency on the Anthropic SDK**, it only builds the dict
shapes the SDK already uses, so the governance layer stays zero-dep and the seam
stays auditable.
"""

from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ._adjudicate import _INVALID_EVIDENCE, _error_detail
from .evidence import compute_verdict

# A PASS verdict allows the action; RETRY and FAIL both block it (RETRY means
# "recoverable, but not as-is", the agent should adjust, same wire signal as FAIL).
Evidence = Iterable[Any]  # Findings or loose evidence dicts; compute_verdict coerces.

#: Shapes that are not a collection of findings. ``None`` and ``False`` are listed so they
#: keep refusing (as they always did), now with a reason that names the shape.
_NOT_EVIDENCE = (type(None), bool, int, float, complex, str, bytes, bytearray)


def _evidence_of(findings: Any) -> List[Any]:
    """Hold the outer evidence container to its contract before the fold sees it.

    The kernel checks each finding (``strict=True``), but an empty invalid container, ``{}``,
    ``""`` or ``b""``, iterates to nothing and would pass as a clean screen. A mapping, a
    string, bytes, a number, a bool or ``None`` is refused here by name; a real empty list
    keeps its meaning (an intentional policy-screen pass), and any other iterable is folded.
    """
    if isinstance(findings, _NOT_EVIDENCE) or isinstance(findings, Mapping):
        raise TypeError(f"evidence was {type(findings).__name__}; pass a list of findings")
    try:
        return list(findings)
    except TypeError:
        raise TypeError(
            f"evidence was {type(findings).__name__}, which is not iterable; pass a list of "
            "findings"
        ) from None


def gate_tool_use(
    tool_use_id: str,
    findings: Evidence,
    *,
    tool_name: str = "tool",
    safe_errors: bool = False,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Adjudicate a proposed Claude tool call against collected evidence.

    Call this in a manual agent loop *before* executing each ``tool_use`` block.
    ``findings`` is whatever evidence you gathered about the proposed call
    (policy checks, preconditions, dry-run results) in the ``compute_verdict``
    finding shape.

    Returns ``(allow, refusal_block)``:
      * ``(True, None)``, verdict is PASS; execute the tool.
      * ``(False, block)``, verdict is RETRY/FAIL; do **not** execute. Append
        ``block`` (a ``tool_result`` with ``is_error: true``) to your next user
        message so Claude sees the refusal and adapts.

    Example (manual loop)::

        for tool in (b for b in resp.content if b.type == "tool_use"):
            allow, refusal = gate_tool_use(tool.id, gather_evidence(tool), tool_name=tool.name)
            if not allow:
                results.append(refusal)
                continue
            results.append({"type": "tool_result", "tool_use_id": tool.id,
                            "content": execute_tool(tool.name, tool.input)})

    Evidence that cannot be folded refuses the call, and by default the refusal quotes
    the error. Pass ``safe_errors=True`` to put a stable code and the exception's class
    name there instead (``RECUSAL_INVALID_EVIDENCE``), so nothing the evidence carried
    reaches the model through an error message.
    """
    # strict at the enforcement boundary: ambiguous evidence (a dict with no
    # status/passed) fails closed to a refusal rather than degrading to PASS.
    try:
        verdict = compute_verdict(_evidence_of(findings), strict=True)
    except Exception as exc:  # noqa: BLE001
        shown = _error_detail(exc, code=_INVALID_EVIDENCE, safe_errors=safe_errors)
        detail = f"invalid evidence ({shown})"
    else:
        if verdict.passed:
            return True, None
        detail = f"[{verdict.decision.value}]: {verdict.reasons()}"

    refusal_block = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": (
            f"Refused by the gate before `{tool_name}` ran {detail}. "
            f"Address the findings and propose a corrected call."
        ),
        "is_error": True,
    }
    return False, refusal_block


def tool_confirmation(
    tool_use_id: str,
    findings: Evidence,
    *,
    safe_errors: bool = False,
) -> Dict[str, Any]:
    """Build a Managed Agents ``user.tool_confirmation`` event from a verdict.

    For agents running ``permission_policy: {"type": "always_ask"}``: when the
    session idles awaiting confirmation of a tool call, adjudicate the evidence
    and send this event back. PASS → ``allow``; RETRY/FAIL → ``deny`` with the
    verdict message as ``deny_message`` (which the agent receives and reacts to).
    ``safe_errors`` is :func:`gate_tool_use`'s.
    """
    try:
        verdict = compute_verdict(
            _evidence_of(findings), strict=True
        )  # strict: fail closed on ambiguous evidence
    except Exception as exc:  # noqa: BLE001
        return {
            "type": "user.tool_confirmation",
            "tool_use_id": tool_use_id,
            "result": "deny",
            "deny_message": (
                "invalid evidence ("
                f"{_error_detail(exc, code=_INVALID_EVIDENCE, safe_errors=safe_errors)})"
            ),
        }
    allow = verdict.passed

    event: Dict[str, Any] = {
        "type": "user.tool_confirmation",
        "tool_use_id": tool_use_id,
        "result": "allow" if allow else "deny",
    }
    if not allow:
        event["deny_message"] = verdict.reasons()
    return event
