"""Prove the INSTALLED recusal-openai-agents-mcp wheel gates MCP calls on a real SDK agent.

Run it from a directory outside the repository, with an interpreter that has the built
wheels installed (recusal, recusal-openai-agents and recusal-openai-agents-mcp):

    python /path/to/adapters/openai-agents-mcp/smoke_test.py

It fails if a package it imports is the source tree and not an installed distribution:
a check run from the checkout would prove nothing about the wheels. An MCP server that
lives in this process, the SDK's scripted model and a scripted approval request stand in
for the real ones; no API key, process or network is used.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any, Optional

# Python puts this script's directory first on the import path. Drop it, so nothing in
# the source tree can stand in for the installed package.
_HERE = Path(__file__).resolve().parent
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != _HERE]

import recusal_openai_agents  # noqa: E402
import recusal_openai_agents_mcp  # noqa: E402
from agents import Agent, HostedMCPTool, RunConfig, Runner  # noqa: E402
from agents.mcp import MCPServer  # noqa: E402
from agents.testing import ScriptedModel, assistant_message, function_call  # noqa: E402
from mcp.types import (  # noqa: E402
    CallToolResult,
    GetPromptResult,
    ListPromptsResult,
    TextContent,
    Tool,
)
from openai.types.responses.response_output_item import McpApprovalRequest  # noqa: E402
from recusal_openai_agents_mcp import (  # noqa: E402
    SURFACES,
    audited_agent,
    gated_agent,
    is_gated,
    is_gated_mcp_server,
)

import recusal  # noqa: E402
from recusal import Finding  # noqa: E402
from recusal.audit import AuditLog, verify  # noqa: E402

CALLED: list[str] = []
CALLS = [("c1", "SELECT 1"), ("c2", "DELETE FROM t")]


def policy(tool_name, tool_input):
    if str(tool_input.get("sql", "")).upper().startswith("DELETE"):
        return [Finding.fail("no_deletes", severity="CRITICAL", message="DELETE is not allowed")]
    return []


class InProcessMCPServer(MCPServer):
    """An MCP server that lives in this process and records every call it is asked to run."""

    @property
    def name(self) -> str:
        return "db"

    async def connect(self) -> None:
        pass

    async def cleanup(self) -> None:
        pass

    async def list_tools(self, run_context: Any = None, agent: Any = None) -> list[Tool]:
        return [
            Tool(
                name="run_sql",
                description="Record SQL text without connecting to a database.",
                inputSchema={
                    "type": "object",
                    "properties": {"sql": {"type": "string"}},
                    "required": ["sql"],
                },
            )
        ]

    async def call_tool(
        self, tool_name: str, arguments: Optional[dict[str, Any]], meta: Any = None
    ) -> CallToolResult:
        sql = dict(arguments or {}).get("sql")
        CALLED.append(sql)
        return CallToolResult(content=[TextContent(type="text", text=f"recorded: {sql}")])

    async def list_prompts(self) -> ListPromptsResult:
        return ListPromptsResult(prompts=[])

    async def get_prompt(
        self, name: str, arguments: Optional[dict[str, Any]] = None
    ) -> GetPromptResult:
        return GetPromptResult(messages=[])


def run(agent: Agent) -> None:
    asyncio.run(Runner.run(agent, "go", run_config=RunConfig(tracing_disabled=True)))


def local(gate) -> str:
    """Propose the calls to an agent whose tools come from a fresh local server, passed
    through ``gate``; return the text of everything the model received back."""
    CALLED.clear()
    turn = [
        function_call("run_sql", json.dumps({"sql": sql}), call_id=ident) for ident, sql in CALLS
    ]
    model = ScriptedModel([turn, [assistant_message("done")]])
    run(gate(Agent(name="assistant", model=model, mcp_servers=[InProcessMCPServer()])))
    call = model.last_call
    assert call is not None and not isinstance(call.input, str)
    return json.dumps(
        [i for i in call.input if isinstance(i, dict) and i.get("type") == "function_call_output"]
    )


def hosted(gate) -> dict[str, dict[str, Any]]:
    """Script the approval requests OpenAI would send; return the SDK's responses by id."""
    requests = [
        McpApprovalRequest(
            id=ident,
            type="mcp_approval_request",
            name="run_sql",
            arguments=json.dumps({"sql": sql}),
            server_label="db",
        )
        for ident, sql in CALLS
    ]
    model = ScriptedModel([requests, [assistant_message("done")]])
    tool = HostedMCPTool(
        tool_config={
            "type": "mcp",
            "server_label": "db",
            "server_url": "https://example.invalid/mcp",
            "require_approval": "always",
        },
        on_approval_request=lambda request: {"approve": True},
    )
    run(gate(Agent(name="assistant", model=model, tools=[tool])))
    call = model.last_call
    assert call is not None and not isinstance(call.input, str)
    return {
        item["approval_request_id"]: item
        for item in call.input
        if isinstance(item, dict) and item.get("type") == "mcp_approval_response"
    }


def main() -> None:
    for module in (recusal, recusal_openai_agents, recusal_openai_agents_mcp):
        origin = Path(module.__file__).resolve()
        assert "site-packages" in origin.parts, f"{module.__name__} is not installed: {origin}"
    assert metadata.version("recusal-openai-agents-mcp") == recusal_openai_agents_mcp.__version__
    requires = metadata.requires("recusal-openai-agents-mcp") or []
    assert any(r.startswith("recusal-openai-agents") for r in requires), requires
    assert any(r.startswith("recusal") and "openai" not in r for r in requires), requires

    # Negative control: without the gate the server is asked to run the DELETE, and the
    # hosted DELETE is approved, so the refusals below are the gate's.
    local(lambda agent: agent)
    assert sorted(CALLED) == ["DELETE FROM t", "SELECT 1"], CALLED
    assert hosted(lambda agent: agent)["c2"]["approve"] is True

    server = InProcessMCPServer()
    assert not is_gated_mcp_server(server)
    assert is_gated(gated_agent(Agent(name="a", mcp_servers=[server]), policy))
    assert is_gated_mcp_server(server), "a local server is gated in place"

    outputs = local(lambda agent: gated_agent(agent, policy))
    assert CALLED == ["SELECT 1"], CALLED
    assert "recorded: SELECT 1" in outputs and "DELETE is not allowed" in outputs, outputs

    responses = hosted(lambda agent: gated_agent(agent, policy))
    assert responses["c1"]["approve"] is True, responses
    assert responses["c2"]["approve"] is False, responses
    assert "DELETE is not allowed" in responses["c2"]["reason"], responses

    with tempfile.TemporaryDirectory() as folder:
        log = AuditLog(path=os.path.join(folder, "audit.jsonl"))
        local(lambda agent: audited_agent(agent, policy, log))
        hosted(lambda agent: audited_agent(agent, policy, log))
        assert CALLED == ["SELECT 1"], CALLED
        actions = [entry["action"] for entry in log.entries]
        on_server = [a for a in actions if a["surface"] == SURFACES["mcp_server"]]
        on_hosted = {
            a.get("tool_use_id"): a for a in actions if a["surface"] == SURFACES["hosted_mcp"]
        }
        assert len(on_server) == 2 and len(on_hosted) == 2, actions
        assert sorted(a["decision"] == "deny" for a in on_server) == [False, True], on_server
        # the SDK hands call_tool no id, so these entries carry none
        assert all(a.get("tool_use_id") is None for a in on_server), on_server
        assert all(a["control"]["mcp_server"] == "db" for a in on_server), on_server
        assert on_hosted["c2"]["decision"] == "deny" and on_hosted["c1"]["decision"] != "deny"
        intact, problems = verify(log.entries)
        assert intact, problems

    print(
        f"recusal-openai-agents-mcp {recusal_openai_agents_mcp.__version__} with "
        f"recusal-openai-agents {recusal_openai_agents.__version__} and recusal "
        f"{recusal.__version__}: installed wheels gate and record"
    )


if __name__ == "__main__":
    main()
