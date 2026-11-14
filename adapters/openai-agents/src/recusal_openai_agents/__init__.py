"""Recusal for the OpenAI Agents SDK: judge every function tool call before it runs.

    from agents import Agent
    from recusal_openai_agents import gated_agent

    agent = gated_agent(Agent(name="assistant", tools=[run_sql]), policy)

``policy(tool_name, tool_input)`` is an ordinary Recusal policy returning findings. A
refused call's output is Recusal's reason, and its tool does not run. ``gate_tools`` gates
a list of tools, and ``audited_agent`` and ``audited_tools`` put each decision on a
hash-chained audit log.

The decision is made by ``recusal.decide``. This package adds only what is the SDK's:
where the guardrail attaches, how the invocation is prepared, and how a refusal is answered.
"""

from ._audit import SURFACE, audited_agent, audited_tools, recording
from ._gate import (
    GUARDRAIL_NAME,
    Policy,
    Recording,
    gate_tools,
    gated_agent,
    is_gated,
    recusal_guardrail,
)

# Building blocks for a gate on another seam of the same SDK (recusal-openai-agents-mcp
# gates MCP servers with them): the decision for a call whose arguments are a JSON
# string, the recorded refusal for a call that cannot be judged, and the tool names an
# agent lists more than once.
from ._gate import _duplicate_names as duplicate_names
from ._gate import _refusal as refusal
from ._gate import _unjudgeable as unjudgeable

__version__ = "0.1.0"

__all__ = [
    "GUARDRAIL_NAME",
    "SURFACE",
    "Policy",
    "Recording",
    "__version__",
    "audited_agent",
    "audited_tools",
    "duplicate_names",
    "gate_tools",
    "gated_agent",
    "is_gated",
    "recording",
    "recusal_guardrail",
    "refusal",
    "unjudgeable",
]
