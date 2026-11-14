"""The audited gate: the same decisions, with every call on a hash-chained ``AuditLog``.

``audited_agent(agent, policy, log)`` is ``gated_agent`` with a ``recording`` that passes
the recording arguments through to ``recusal.decide``, which writes the entry before
returning. There is no second decision path. If the log cannot be written the decision
becomes a deny, so nothing runs unrecorded. One log covers the agent's three seams, each
under its own surface label:

- ``openai_agents.tool_input_guardrail``: function tools, under the SDK's tool call id;
- ``openai_agents.mcp.call_tool``: a local MCP server's ``call_tool``. The SDK hands that
  method the server's own tool name and the arguments, and no call id, so these entries
  carry none. The server's name is recorded as ``control.mcp_server``, and the entry is
  bound to the arguments the server receives;
- ``openai_agents.hosted_mcp.approval``: a hosted MCP call OpenAI asked approval for, under
  the approval request's id, with the server label as ``control.mcp_server``, bound to the
  arguments in the request.

What is not on the log: a hosted call OpenAI does not ask approval for (the gate never
sees it, which is why the gate sets ``require_approval`` to ``"always"``), and what the
tool's own approval callback answered after Recusal allowed a call: the entry records
Recusal's decision, not that callback's.

A local server is a shared connection and carries one policy and one recording: gating it
again with a different log is refused, as a different policy is.
"""

from __future__ import annotations

from typing import Any, Optional

from agents import Agent

from recusal.audit import AuditLog

from ._gate import Policy, Recording, gated_agent

#: The surface label per seam: which gate of the agent adjudicated the call.
SURFACES = {
    "function_tool": "openai_agents.tool_input_guardrail",
    "mcp_server": "openai_agents.mcp.call_tool",
    "hosted_mcp": "openai_agents.hosted_mcp.approval",
}


def _text_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def recording(
    log: AuditLog, *, actor: Optional[str] = None, control: Optional[dict[str, Any]] = None
) -> Recording:
    """The recording arguments for each call, built in one place for the three seams.

    The call id is the SDK's own where the seam has one: the tool call id for a function
    tool, the approval request id for hosted MCP, none for a local server's ``call_tool``.
    The MCP server a call went to is recorded as ``control.mcp_server``; a caller's own
    ``control`` cannot overwrite it."""

    def for_call(seam: str, context: Any) -> dict[str, Any]:
        identity = dict(control or {})
        ident: Optional[str] = None
        if seam == "function_tool":
            ident = _text_or_none(getattr(context, "tool_call_id", None))
        elif seam == "hosted_mcp":
            data = getattr(context, "data", None)
            ident = _text_or_none(getattr(data, "id", None))
            identity["mcp_server"] = getattr(data, "server_label", None)
        else:
            identity["mcp_server"] = getattr(context, "name", None)
        return {
            "audit": log,
            "surface": SURFACES[seam],
            "tool_use_id": ident,
            "actor": actor,
            "control": identity or None,
        }

    return for_call


def audited_agent(
    agent: Agent,
    policy: Policy,
    log: AuditLog,
    *,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
) -> Agent:
    """``gated_agent`` with every decision, on function tools, local MCP servers and hosted
    MCP approvals, on ``log``."""
    return gated_agent(agent, policy, recording(log, actor=actor, control=control))
