"""Recusal for CrewAI: judge every tool call before it runs.

    from recusal_crewai import install_recusal_hooks

    uninstall = install_recusal_hooks(policy)   # after your other hooks
    crew.kickoff()

``policy(tool_name, tool_input)`` is an ordinary Recusal policy returning findings. A
refused call is blocked, and the model reads Recusal's reason in place of CrewAI's fixed
"blocked by hook" line. CrewAI's hook registry is process-global, so one install covers
every agent, task and crew in the process. ``install_audited_hooks`` puts each decision on
a hash-chained audit log.

The decision is made by ``recusal.decide``. This package adds only what is CrewAI's: the
two hooks, how the invocation is prepared, and how a refusal is explained to the model.
"""

from ._audit import SURFACE, install_audited_hooks, recording
from ._gate import (
    BLOCKED_PREFIX,
    PENDING_LIMIT,
    Policy,
    Recording,
    install_recusal_hooks,
    recusal_after_tool_call,
    recusal_before_tool_call,
)

__version__ = "0.1.0"

__all__ = [
    "BLOCKED_PREFIX",
    "PENDING_LIMIT",
    "SURFACE",
    "Policy",
    "Recording",
    "__version__",
    "install_audited_hooks",
    "install_recusal_hooks",
    "recording",
    "recusal_after_tool_call",
    "recusal_before_tool_call",
]
