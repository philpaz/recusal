"""
Offline OpenAI Agents SDK example: gate MCP tools, local servers and hosted MCP, with Recusal.

Requires Python 3.10+ and is tested with exactly openai-agents==0.24.0 and openai==3.34.0.
Recusal itself remains Python 3.9+ and has no OpenAI dependency. The function-tool gate it
builds on is examples/openai_agents_gate.py; this example covers the two MCP paths that one
names as out of its sight.

Local MCP servers (``Agent(mcp_servers=[...])``). The SDK connects to these from your own
process, lists each server's tools, and gives every tool a public name the model calls:
the tool's own name by default, or under ``mcp_config={"include_server_in_tool_names":
True}`` a prefixed form, sanitized to the characters the model API accepts, hashed when two
names collide or a function tool already holds the name, and shortened when too long. A
gate keyed on that public name judges a presentation of the capability, not the
capability. The seam this example uses is the one place the SDK hands the server the
capability itself: ``server.call_tool(tool_name, arguments)``, invoked with the server's own
tool name and the exact argument dict the server will run, after any approval a person
gave. ``gate_mcp_server(server, policy)`` wraps that method on the server, in place, because
a server object is a connection and a copy would be a second one. A refusal returns an MCP
error result carrying Recusal's reason, which the SDK surfaces to the model as the call's
output, and the server's own ``call_tool`` is never reached. The policy therefore sees the
name the server executes under, whatever the model called it, and a policy keyed by tool
name must name the server's own tools.

Hosted MCP (``HostedMCPTool``). Those calls run on OpenAI's servers, and the one local point
is the SDK's approval callback, which runs only for a call OpenAI asks approval for.
``gate_hosted_mcp(tool, policy)`` returns a copy whose ``require_approval`` is ``"always"``
and whose ``on_approval_request`` is ``recusal_approval(policy)``: a refusal answers
``{"approve": False, "reason": ...}``, which the SDK sends back as the approval response,
so the hosted call is not made and the model reads the reason. An existing callback keeps
running for the calls the gate allows. This is approval-gated, not execution-gated: a call
OpenAI does not ask approval for is never seen, so a hosted tool whose ``require_approval``
is anything but ``"always"`` is reported as ungated.

``gated_agent(agent, policy)`` gates all three at once: function tools (through the
function-tool example's ``gate_tools``), every server in ``mcp_servers`` in place, and every
``HostedMCPTool`` as a gated copy. It refuses, rather than skips, what no gate can see:
handoffs, a sub-agent behind ``as_tool`` that is not itself gated, and ``Agent`` subclasses.

A server carries one policy. Because the gate is installed on the server object, every
agent that lists that object shares it: gating it again with the same policy is a no-op,
and gating it with a different policy is refused with a ``ValueError`` rather than kept or
replaced silently. Use a separate server object per policy.

No model, API key, process or network is used: ``agents.testing.ScriptedModel`` proposes
the calls, ``InProcessMCPServer`` is an MCP server that lives in this process and records
what it was asked to run, and a scripted approval request stands in for OpenAI's.

Run after installing:
    pip install openai-agents==0.24.0 openai==3.34.0
    python examples/openai_agents_mcp_gate.py
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import os
import sys
from typing import Any, Callable, List, Optional

if sys.version_info < (3, 10):
    print(
        "openai_agents_mcp_gate.py requires Python 3.10+ (openai-agents 0.24.0); "
        f"this is Python {sys.version_info[0]}.{sys.version_info[1]}.",
        file=sys.stderr,
    )
    raise SystemExit(2)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.openai_agents_gate import (  # noqa: E402
    OPENAI_AGENTS_VERSION,
    OPENAI_VERSION,
    Agent,
    RunConfig,
    Runner,
    ScriptedModel,
    _duplicate_names,
    _refusal,
    assistant_message,
    gate_tools,
    scripted_model,
    sql_scope_policy,
    tool_outputs,
)
from examples.openai_agents_gate import (  # noqa: E402
    is_gated as function_tools_gated,
)
from recusal.claude_code import _identity_getter, decide  # noqa: E402

try:
    from agents import HostedMCPTool, MCPToolApprovalRequest
    from agents.mcp import MCPServer
    from mcp.types import CallToolResult, GetPromptResult, ListPromptsResult, TextContent, Tool
    from openai.types.responses.response_output_item import McpApprovalRequest
except ImportError as exc:
    raise SystemExit(
        "Install the example dependencies first: "
        f"openai-agents=={OPENAI_AGENTS_VERSION} openai=={OPENAI_VERSION}"
    ) from exc

Policy = Callable[[str, dict], List[Any]]
CALLED: list[tuple[str, str, dict[str, Any]]] = []
"""Every (server, tool, arguments) an in-process MCP server was asked to run."""


# --- local MCP servers -----------------------------------------------------------------------


def _refusal_for_call(tool_name: str, arguments: Any, policy: Policy) -> Optional[str]:
    """Return the reason a server call is refused, or None to let it through.

    ``arguments`` is the dict the SDK hands ``call_tool`` (``None`` for a call without
    any), so the policy judges exactly what the server will receive. A non-mapping is
    refused: the server would not accept it either, and the policy cannot read it.

    The policy judges a deep copy of that dict (JSON data, so the copy is exact) and the
    server receives the dict itself, so a policy that writes to what it was handed, after
    gathering its evidence, changes its copy and not the call.
    """
    args = {} if arguments is None else arguments
    if not isinstance(args, dict):
        return f"Recusal refused `{tool_name}`: arguments are not a mapping"
    try:
        decision, reason = decide(tool_name, copy.deepcopy(args), policy, safe_errors=True)
    except Exception as exc:  # noqa: BLE001, a buggy policy must not silently disable the gate
        return f"Recusal failed closed (policy error): RECUSAL_POLICY_ERROR ({type(exc).__name__})"
    return reason if decision == "deny" else None


def claude_named(server_name: str, policy: Policy) -> Policy:
    """Bind a policy written for Claude Code's MCP tool names to one server connection.

    :func:`recusal.mcp.manifest_policy` (and any composition carrying one) judges
    ``mcp__<server>__<tool>``, the name Claude Code gives an MCP tool. The gates in this
    module hand a policy the server's own tool name, so that policy would never see an MCP
    name, run no membership check and raise no objection. The bound policy presents each
    call as ``mcp__{server_name}__{tool}``, carries the server it is bound to (a gate refuses
    it for any other server, and so does a composition carrying it), and forwards the
    manifest digest to the audit record through the policy's own control identity. A policy
    already bound to this server is returned as it is, so binding twice never prefixes
    twice; one bound to another server is refused.
    """
    if not isinstance(server_name, str) or not server_name:
        raise ValueError("claude_named needs the name of the server the policy is bound to")
    already = getattr(policy, "recusal_bound_server", None)
    if already is not None:
        if already != server_name:
            raise ValueError(
                f"Recusal: this policy is already bound to server {already!r} by claude_named, "
                f"not to {server_name!r}; bind one policy object per server"
            )
        return policy

    def bound(tool_name: str, arguments: dict) -> List[Any]:
        return policy(f"mcp__{server_name}__{tool_name}", arguments)

    bound.recusal_bound_server = server_name  # type: ignore[attr-defined]
    bound.recusal_bound_policy = policy  # type: ignore[attr-defined]
    getter = _identity_getter(policy)
    if getter is not None:
        bound.get_control_identity = getter  # type: ignore[attr-defined]
    reset = getattr(policy, "reset_control_identity", None)
    if callable(reset):
        bound.reset_control_identity = reset  # type: ignore[attr-defined]
    return bound


def _name_domain_refusal(policy: Policy, server_name: Any, where: str) -> Optional[str]:
    """Why this policy cannot judge this server's own tool names, or None if it can."""
    # The naming requirement is checked before the binding: a policy that still judges
    # Claude Code's names is never installable, bound neighbour or not.
    if getattr(policy, "recusal_name_domain", None) == "claude_code":
        return (
            "Recusal: this policy judges Claude Code's MCP names (`mcp__<server>__<tool>`, the "
            f"shape of recusal.mcp.manifest_policy), but {where} hands a policy the server's "
            "own tool name, so its membership check would see no MCP name and raise no "
            f"objection. Bind it to the connection: claude_named({server_name!r}, policy)."
        )
    bound_to = getattr(policy, "recusal_bound_server", None)
    if bound_to is not None and bound_to != server_name:
        return (
            f"Recusal: this policy is bound to server {bound_to!r} by claude_named, not to "
            f"{server_name!r}; bind one policy object per server"
        )
    return None


def gate_mcp_server(server: MCPServer, policy: Policy) -> MCPServer:
    """Wrap the server's ``call_tool`` so every call is judged by the server's own tool
    name and the exact arguments it would run, in place; returns the same server.

    The SDK invokes ``call_tool`` with the raw tool name whatever public name the model
    used, after any approval, so this is the point where the policy judges the capability
    that executes. A refusal is returned as an MCP error result carrying the reason, and
    the server's own ``call_tool`` is not reached.
    """
    if not isinstance(server, MCPServer):
        raise TypeError(f"Recusal cannot gate {type(server).__name__}: not an agents.mcp.MCPServer")
    mismatch = _name_domain_refusal(policy, server.name, "gate_mcp_server")
    if mismatch is not None:
        raise ValueError(mismatch)
    if is_gated_mcp_server(server):
        # A server object is a connection, shared by every agent that lists it, and it
        # carries ONE policy. Installing the same policy again is a no-op; asking for a
        # different one is refused, because silently keeping the first would leave the
        # caller believing a stricter policy took effect, and silently replacing it would
        # change the policy under every other agent sharing the connection.
        if server.call_tool.recusal_policy is policy:  # type: ignore[attr-defined]
            return server
        raise ValueError(
            f"Recusal: server {server.name!r} is already gated with a different policy; a "
            "connection carries one policy for every agent that shares it. Gate a separate "
            "server object for a different policy."
        )
    original = server.call_tool

    async def gated_call_tool(
        tool_name: str, arguments: Optional[dict[str, Any]], meta: Any = None
    ) -> CallToolResult:
        reason = _refusal_for_call(tool_name, arguments, policy)
        if reason is not None:
            return CallToolResult(content=[TextContent(type="text", text=reason)], isError=True)
        if meta is None:
            return await original(tool_name, arguments)
        return await original(tool_name, arguments, meta=meta)

    gated_call_tool.recusal_gate = True  # type: ignore[attr-defined]
    gated_call_tool.recusal_policy = policy  # type: ignore[attr-defined]
    server.call_tool = gated_call_tool  # type: ignore[method-assign]
    return server


def is_gated_mcp_server(server: Any) -> bool:
    """True if the server's ``call_tool`` is wrapped by the Recusal gate."""
    return isinstance(server, MCPServer) and bool(getattr(server.call_tool, "recusal_gate", False))


# --- hosted MCP -------------------------------------------------------------------------------


def recusal_approval(policy: Policy, then: Any = None):
    """Return an ``on_approval_request`` callback that judges each hosted call OpenAI asks about.

    A refusal answers ``{"approve": False, "reason": reason}``. An allowed call is passed to
    ``then`` (the callback the tool already had), or approved when there is none.
    """

    async def gate(request: MCPToolApprovalRequest) -> dict:
        reason = _refusal(request.data.name, request.data.arguments, policy)
        if reason is not None:
            return {"approve": False, "reason": reason}
        if then is None:
            return {"approve": True}
        result = then(request)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    gate.recusal_gate = True  # type: ignore[attr-defined]
    return gate


def gate_hosted_mcp(tool: HostedMCPTool, policy: Policy) -> HostedMCPTool:
    """Return a copy of the hosted MCP tool that requires approval for every call and judges
    each approval request with Recusal; the original is unchanged."""
    if not isinstance(tool, HostedMCPTool):
        raise TypeError(f"Recusal cannot gate {type(tool).__name__}: not a HostedMCPTool")
    mismatch = _name_domain_refusal(policy, tool.tool_config.get("server_label"), "gate_hosted_mcp")
    if mismatch is not None:
        raise ValueError(mismatch)
    config = {**tool.tool_config, "require_approval": "always"}
    return dataclasses.replace(
        tool,
        tool_config=config,
        on_approval_request=recusal_approval(policy, then=tool.on_approval_request),
    )


def is_gated_hosted_mcp(tool: Any) -> bool:
    """True if every call to this hosted tool is sent for approval and judged by Recusal."""
    return (
        isinstance(tool, HostedMCPTool)
        and tool.tool_config.get("require_approval") == "always"
        and getattr(tool.on_approval_request, "recusal_gate", False)
    )


# --- the whole agent --------------------------------------------------------------------------


def _ungateable_reason(agent: Any) -> Optional[str]:
    if type(agent) is not Agent:
        return (
            f"{type(agent).__name__} is not a plain Agent; a subclass can add tools at run "
            "time that no gate sees"
        )
    if agent.handoffs:
        return "it has handoffs, which hand the conversation to another agent's tools"
    duplicates = _duplicate_names(agent.tools)
    if duplicates:
        return (
            f"its function tools share a name {duplicates!r}; the SDK resolves a shared name "
            "last-wins at execution, so a guardrail could judge one tool while another runs"
        )
    return None


def is_gated(agent: Any) -> bool:
    """True if every call this agent can make, function tool, local MCP tool or hosted MCP
    call OpenAI asks approval for, passes through a Recusal gate."""
    if _ungateable_reason(agent) is not None:
        return False
    hosted = [tool for tool in agent.tools if isinstance(tool, HostedMCPTool)]
    others = [tool for tool in agent.tools if not isinstance(tool, HostedMCPTool)]
    if not all(is_gated_hosted_mcp(tool) for tool in hosted):
        return False
    if not all(is_gated_mcp_server(server) for server in agent.mcp_servers):
        return False
    return function_tools_gated(agent.clone(tools=others, mcp_servers=[]))


def gated_agent(agent: Agent, policy: Policy) -> Agent:
    """Return a copy of the agent whose every tool call is judged by Recusal.

    Function tools and hosted MCP tools are replaced by gated copies; the servers in
    ``mcp_servers`` are gated in place and shared with the original. Raises ValueError if
    the agent can run something no gate sees.
    """
    reason = _ungateable_reason(agent)
    if reason is not None:
        raise ValueError(f"Recusal cannot gate agent {agent.name!r}: {reason}")
    tools: list[Any] = []
    for tool in agent.tools:
        if isinstance(tool, HostedMCPTool):
            tools.append(gate_hosted_mcp(tool, policy))
        else:
            tools.extend(gate_tools([tool], policy))
    servers = [gate_mcp_server(server, policy) for server in agent.mcp_servers]
    return agent.clone(tools=tools, mcp_servers=servers)


# --- offline fixtures: an in-process MCP server, a scripted model, a scripted approval ------


class InProcessMCPServer(MCPServer):
    """An MCP server that lives in this process: lists its tools, records every call."""

    def __init__(
        self, name: str = "db", tools: tuple[str, ...] = ("echo", "run_sql"), **kwargs: Any
    ):
        super().__init__(**kwargs)
        self._name = name
        self._tools = tuple(tools)

    @property
    def name(self) -> str:
        return self._name

    async def connect(self) -> None:
        pass

    async def cleanup(self) -> None:
        pass

    async def list_tools(self, run_context: Any = None, agent: Any = None) -> list[Tool]:
        schemas = {
            "echo": ("Return harmless text.", {"text": {"type": "string"}}),
            "run_sql": (
                "Record SQL text without connecting to a database.",
                {"sql": {"type": "string"}},
            ),
        }
        listed = []
        for name in self._tools:
            description, properties = schemas.get(
                name, ("Record SQL text.", {"sql": {"type": "string"}})
            )
            listed.append(
                Tool(
                    name=name,
                    description=description,
                    inputSchema={
                        "type": "object",
                        "properties": properties,
                        "required": list(properties),
                    },
                )
            )
        return listed

    async def call_tool(
        self, tool_name: str, arguments: Optional[dict[str, Any]], meta: Any = None
    ) -> CallToolResult:
        args = dict(arguments or {})
        CALLED.append((self.name, tool_name, args))
        text = args.get("text", "") if tool_name == "echo" else f"recorded SQL: {args.get('sql')}"
        return CallToolResult(content=[TextContent(type="text", text=text)])

    async def list_prompts(self) -> ListPromptsResult:
        return ListPromptsResult(prompts=[])

    async def get_prompt(
        self, name: str, arguments: Optional[dict[str, Any]] = None
    ) -> GetPromptResult:
        return GetPromptResult(messages=[])


def approval_request(name: str, arguments: Any, request_id: str, server_label: str = "db"):
    """A scripted approval request, as OpenAI sends one for a hosted MCP call."""
    serialized = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return McpApprovalRequest(
        id=request_id,
        type="mcp_approval_request",
        name=name,
        arguments=serialized,
        server_label=server_label,
    )


def approval_responses(model: ScriptedModel) -> dict[str, dict[str, Any]]:
    """Each approval response the SDK sent back, keyed by request id."""
    call = model.last_call
    assert call is not None and not isinstance(call.input, str)
    return {
        item["approval_request_id"]: item
        for item in call.input
        if isinstance(item, dict) and item.get("type") == "mcp_approval_response"
    }


async def run_agent(agent: Agent) -> Any:
    return await Runner.run(agent, "go", run_config=RunConfig(tracing_disabled=True))


def run_calls(
    calls: list[tuple[str, Any, str]],
    *,
    policy: Policy = sql_scope_policy,
    gated: bool = True,
    server: Optional[MCPServer] = None,
) -> tuple[dict[str, Any], list[tuple[str, str, dict[str, Any]]]]:
    """Propose the calls to an agent whose only tools come from a local MCP server; return
    each call's output as the model received it, and what the server was asked to run."""
    CALLED.clear()
    model = scripted_model(calls)
    agent = Agent(name="assistant", model=model, mcp_servers=[server or InProcessMCPServer()])
    if gated:
        agent = gated_agent(agent, policy)
    asyncio.run(run_agent(agent))
    return tool_outputs(model), list(CALLED)


def run_hosted(
    requests: list[tuple[str, Any, str]],
    *,
    policy: Policy = sql_scope_policy,
    gated: bool = True,
    tool: Optional[HostedMCPTool] = None,
) -> dict[str, dict[str, Any]]:
    """Script the approval requests OpenAI would send for hosted calls; return the SDK's
    approval responses by request id."""
    model = ScriptedModel(
        [
            [
                approval_request(name, arguments, request_id)
                for name, arguments, request_id in requests
            ],
            [assistant_message("done")],
        ]
    )
    hosted = tool or HostedMCPTool(
        tool_config={
            "type": "mcp",
            "server_label": "db",
            "server_url": "https://example.invalid/mcp",
            "require_approval": "always",
        },
        on_approval_request=lambda request: {"approve": True},
    )
    agent = Agent(name="assistant", model=model, tools=[hosted])
    if gated:
        agent = gated_agent(agent, policy)
    asyncio.run(run_agent(agent))
    return approval_responses(model)


def _text(output: Any) -> str:
    if isinstance(output, list):
        return "".join(item.get("text", "") for item in output if isinstance(item, dict))
    return str(output)


def main() -> None:
    calls = [
        ("echo", {"text": "hello"}, "c1"),
        ("run_sql", {"sql": "DELETE FROM audit_log"}, "c2"),
        ("run_sql", {"sql": "SELECT * FROM audit_log"}, "c3"),
    ]
    outputs, called = run_calls(calls)
    print("local MCP server:")
    for name, _, call_id in calls:
        print(f"  {name} ({call_id}): {_text(outputs[call_id])}")
    print(f"  tools the server ran: {len(called)}")
    responses = run_hosted([(name, args, f"r{i}") for i, (name, args, _) in enumerate(calls, 1)])
    print("hosted MCP, approval requests:")
    for name, _, request_id in [(n, a, f"r{i}") for i, (n, a, _) in enumerate(calls, 1)]:
        response = responses[request_id]
        verdict = "approved" if response["approve"] else f"rejected: {response['reason']}"
        print(f"  {name} ({request_id}): {verdict}")


if __name__ == "__main__":
    main()
