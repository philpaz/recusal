"""Recusal for MCP tools on an OpenAI Agents SDK agent: local servers and hosted MCP.

    from agents import Agent
    from recusal_openai_agents_mcp import gated_agent

    agent = gated_agent(Agent(name="assistant", tools=[...], mcp_servers=[server]), policy)

``policy(tool_name, tool_input)`` is an ordinary Recusal policy returning findings.
``gated_agent`` gates the agent's function tools, every local MCP server in place, and
every hosted MCP tool as a copy that always asks approval. ``gate_mcp_server`` and
``gate_hosted_mcp`` gate one of each, and ``audited_agent`` puts each decision on a
hash-chained audit log.

The decision is made by ``recusal.decide``, and the function-tool gate is
``recusal-openai-agents``. This package adds only the two MCP seams.
"""

from ._audit import SURFACES, audited_agent, recording
from ._gate import (
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

__version__ = "0.1.0"

__all__ = [
    "SURFACES",
    "Policy",
    "Recording",
    "__version__",
    "audited_agent",
    "claude_named",
    "gate_hosted_mcp",
    "gate_mcp_server",
    "gated_agent",
    "is_gated",
    "is_gated_hosted_mcp",
    "is_gated_mcp_server",
    "recording",
    "recusal_approval",
]
