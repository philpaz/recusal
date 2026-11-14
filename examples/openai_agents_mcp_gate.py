"""
Offline OpenAI Agents SDK example: gate MCP tools, local servers and hosted MCP, with Recusal.

The gate is the ``recusal-openai-agents-mcp`` package (``adapters/openai-agents-mcp`` in
this repository): ``gated_agent``, ``gate_mcp_server``, ``gate_hosted_mcp`` and the rest
are imported from it. This file is the offline demonstration around it: an MCP server
that lives in this process, a scripted model and a scripted approval request.

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
import json
import os
import sys
from typing import Any, Optional

if sys.version_info < (3, 10):
    print(
        "openai_agents_mcp_gate.py requires Python 3.10+ (openai-agents 0.24.0); "
        f"this is Python {sys.version_info[0]}.{sys.version_info[1]}.",
        file=sys.stderr,
    )
    raise SystemExit(2)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
# In a checkout the adapters are read from the tree; installed, `pip install
# recusal-openai-agents-mcp` provides the same packages and these paths do not exist.
sys.path.insert(0, os.path.join(_ROOT, "adapters", "openai-agents", "src"))
sys.path.insert(0, os.path.join(_ROOT, "adapters", "openai-agents-mcp", "src"))

from recusal_openai_agents_mcp import (  # noqa: E402
    Policy,
    Recording,
    claude_named,
    gate_hosted_mcp,
    gate_mcp_server,
    gated_agent,
    is_gated,
    is_gated_hosted_mcp,
    is_gated_mcp_server,
    recusal_approval,
)

from examples.openai_agents_gate import (  # noqa: E402
    OPENAI_AGENTS_VERSION,
    OPENAI_VERSION,
    Agent,
    RunConfig,
    Runner,
    ScriptedModel,
    assistant_message,
    scripted_model,
    sql_scope_policy,
    tool_outputs,
)

try:
    from agents import HostedMCPTool
    from agents.mcp import MCPServer
    from mcp.types import CallToolResult, GetPromptResult, ListPromptsResult, TextContent, Tool
    from openai.types.responses.response_output_item import McpApprovalRequest
except ImportError as exc:
    raise SystemExit(
        "Install the example dependencies first: "
        f"openai-agents=={OPENAI_AGENTS_VERSION} openai=={OPENAI_VERSION}"
    ) from exc

__all__ = [
    "Policy",
    "Recording",
    "claude_named",
    "gate_hosted_mcp",
    "gate_mcp_server",
    "gated_agent",
    "is_gated",
    "is_gated_hosted_mcp",
    "is_gated_mcp_server",
    "recusal_approval",
]

CALLED: list[tuple[str, str, dict[str, Any]]] = []
"""Every (server, tool, arguments) an in-process MCP server was asked to run."""


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
