"""
Offline OpenAI Agents SDK example: gate MCP tools, local servers and hosted MCP, with Recusal.

Requires Python 3.10+ and is tested with exactly openai-agents==0.24.0 and openai==3.34.0.
Recusal itself remains Python 3.9+ and has no OpenAI dependency. The function-tool gate it
builds on is examples/openai_agents_gate.py; this example covers the two MCP paths that one
names as out of its sight.

Local MCP servers (``Agent(mcp_servers=[...])``). The SDK connects to these from your own
process and turns every tool a server lists into a function tool at run time, copying the
server's ``tool_input_guardrails`` onto each one. So the seam is the same tool input
guardrail as the function-tool example: ``gate_mcp_server(server, policy)`` appends
``recusal_guardrail(policy)`` to the server's list, in place, because a server object is a
connection and a copy would be a second one. The guardrail then runs immediately before
each MCP tool executes, after any approval a person gave; a refusal becomes the call's
output, so the model reads Recusal's reason and the server is never called. The policy
sees the server's own tool name, the one the server will run. With
``mcp_config={"include_server_in_tool_names": True}`` the model calls the prefixed public
name ``mcp_<server>__<tool>``, and the gate, which knows its server's name, maps that back
before judging; a public name shortened to a hash (a very long server or tool name) cannot
be mapped and is judged as the model called it.

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

No model, API key, process or network is used: ``agents.testing.ScriptedModel`` proposes
the calls, ``InProcessMCPServer`` is an MCP server that lives in this process and records
what it was asked to run, and a scripted approval request stands in for OpenAI's.

Run after installing:
    pip install openai-agents==0.24.0 openai==3.34.0
    python examples/openai_agents_mcp_gate.py
"""

from __future__ import annotations

import asyncio
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
    GUARDRAIL_NAME,
    OPENAI_AGENTS_VERSION,
    OPENAI_VERSION,
    Agent,
    RunConfig,
    Runner,
    ScriptedModel,
    _refusal,
    assistant_message,
    gate_tools,
    scripted_model,
    sql_scope_policy,
    tool_outputs,
)
from examples.openai_agents_gate import (
    is_gated as function_tools_gated,
)

try:
    from agents import (
        HostedMCPTool,
        MCPToolApprovalRequest,
        ToolGuardrailFunctionOutput,
        tool_input_guardrail,
    )
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


def _carries_gate(guardrails: Any) -> bool:
    return any(
        getattr(guardrail.guardrail_function, "recusal_gate", False)
        for guardrail in guardrails or ()
    )


def _public_prefix(server_name: str) -> str:
    """The prefix the SDK puts on this server's tool names under
    ``include_server_in_tool_names``: ``mcp_<server>__``, the server part reduced to ASCII
    letters, digits, ``_`` and ``-`` (the compat probe pins this against the SDK's own)."""
    safe = "".join(
        char if char.isascii() and (char.isalnum() or char in "_-") else "_" for char in server_name
    ).strip("_-")
    return f"mcp_{safe or 'server'}__"


def _judged_name(public_name: str, server_name: str) -> str:
    """The server's own tool name behind a public name, when it carries this server's prefix."""
    prefix = _public_prefix(server_name)
    if public_name.startswith(prefix) and len(public_name) > len(prefix):
        return public_name[len(prefix) :]
    return public_name


def recusal_mcp_guardrail(policy: Policy, server_name: str):
    """The tool input guardrail for one server: judges each call by the server's own tool
    name and the arguments the server will receive."""

    def gate(data) -> ToolGuardrailFunctionOutput:
        name = _judged_name(data.context.tool_name, server_name)
        reason = _refusal(name, data.context.tool_arguments, policy)
        if reason is not None:
            return ToolGuardrailFunctionOutput.reject_content(reason)
        return ToolGuardrailFunctionOutput.allow()

    gate.recusal_gate = True  # type: ignore[attr-defined]
    return tool_input_guardrail(gate, name=GUARDRAIL_NAME)


def gate_mcp_server(server: MCPServer, policy: Policy) -> MCPServer:
    """Append the Recusal guardrail to the server's tool input guardrails, in place.

    The SDK copies a server's guardrails onto every tool it lists, so one gate on the server
    covers each of its tools in every agent that uses it. Returns the same server.
    """
    if not isinstance(server, MCPServer):
        raise TypeError(f"Recusal cannot gate {type(server).__name__}: not an agents.mcp.MCPServer")
    server.tool_input_guardrails = [
        *(server.tool_input_guardrails or ()),
        recusal_mcp_guardrail(policy, server.name),
    ]
    return server


def is_gated_mcp_server(server: Any) -> bool:
    """True if the server carries the Recusal guardrail."""
    return isinstance(server, MCPServer) and _carries_gate(server.tool_input_guardrails)


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
    """An MCP server that lives in this process: lists echo and run_sql, records every call."""

    def __init__(self, name: str = "db", **kwargs: Any):
        super().__init__(**kwargs)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def connect(self) -> None:
        pass

    async def cleanup(self) -> None:
        pass

    async def list_tools(self, run_context: Any = None, agent: Any = None) -> list[Tool]:
        return [
            Tool(
                name="echo",
                description="Return harmless text.",
                inputSchema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            ),
            Tool(
                name="run_sql",
                description="Record SQL text without connecting to a database.",
                inputSchema={
                    "type": "object",
                    "properties": {"sql": {"type": "string"}},
                    "required": ["sql"],
                },
            ),
        ]

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
