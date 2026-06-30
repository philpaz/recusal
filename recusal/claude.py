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

from typing import Any, Dict, Iterable, Optional, Tuple

from .evidence import compute_verdict

# A PASS verdict allows the action; RETRY and FAIL both block it (RETRY means
# "recoverable, but not as-is", the agent should adjust, same wire signal as FAIL).
Evidence = Iterable[Any]  # Findings or loose evidence dicts; compute_verdict coerces.


def gate_tool_use(
    tool_use_id: str,
    findings: Evidence,
    *,
    tool_name: str = "tool",
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Adjudicate a proposed Claude tool call against collected evidence.

    Call this in a manual agent loop *before* executing each ``tool_use`` block.
    ``findings`` is whatever evidence you gathered about the proposed call
    (policy checks, preconditions, dry-run results) in the ``compute_verdict``
    finding shape.

    Returns ``(allow, refusal_block)``:
      * ``(True, None)`` , verdict is PASS; execute the tool.
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
    """
    # strict at the enforcement boundary: ambiguous evidence (a dict with no
    # status/passed) fails closed to a refusal rather than degrading to PASS.
    try:
        verdict = compute_verdict(findings, strict=True)
    except Exception as exc:  # noqa: BLE001
        detail = f"invalid evidence ({exc})"
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
) -> Dict[str, Any]:
    """Build a Managed Agents ``user.tool_confirmation`` event from a verdict.

    For agents running ``permission_policy: {"type": "always_ask"}``: when the
    session idles awaiting confirmation of a tool call, adjudicate the evidence
    and send this event back. PASS → ``allow``; RETRY/FAIL → ``deny`` with the
    verdict message as ``deny_message`` (which the agent receives and reacts to).
    """
    try:
        verdict = compute_verdict(
            findings, strict=True
        )  # strict: fail closed on ambiguous evidence
    except Exception as exc:  # noqa: BLE001
        return {
            "type": "user.tool_confirmation",
            "tool_use_id": tool_use_id,
            "result": "deny",
            "deny_message": f"invalid evidence ({exc})",
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
