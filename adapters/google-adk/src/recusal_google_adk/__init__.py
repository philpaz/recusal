"""Recusal for Google ADK: judge every tool call before it runs.

    from google.adk.agents import LlmAgent
    from recusal_google_adk import gated_agent

    agent = gated_agent(LlmAgent(name="assistant", model=model, tools=[run_sql]), policy)

``policy(tool_name, tool_input)`` is an ordinary Recusal policy returning findings. A
refused call is answered to the model as ``{"error": reason}`` and its tool does not run.
``RecusalPlugin(policy)`` is the same gate for every agent a runner drives, and
``audited_agent`` and ``audited_plugin`` put each decision on a hash-chained audit log.

The decision is made by ``recusal.decide``. This package adds only what is the ADK's:
where the callback attaches, how the invocation is prepared, and how a refusal is answered.
"""

from ._audit import SURFACE, audited_agent, audited_plugin, recording
from ._gate import (
    Policy,
    Recording,
    RecusalPlugin,
    gated_agent,
    is_gated,
    recusal_before_tool,
)

__version__ = "0.1.0"

__all__ = [
    "SURFACE",
    "Policy",
    "Recording",
    "RecusalPlugin",
    "__version__",
    "audited_agent",
    "audited_plugin",
    "gated_agent",
    "is_gated",
    "recording",
    "recusal_before_tool",
]
