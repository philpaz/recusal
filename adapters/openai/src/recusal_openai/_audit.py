"""The audited loop: the same decision, with every call on a hash-chained ``AuditLog``.

``audited_function_calls`` and ``audited_tool_calls`` call ``gate_function_calls`` and
``gate_tool_calls`` with the recording arguments passed through to ``recusal.decide``,
which writes the entry before returning. There is no second decision path. If the log
cannot be written the decision becomes a deny, so nothing runs unrecorded.

Each entry carries the surface of the API shape that adjudicated the call and the API's own
call id, and a SHA-256 fingerprint of the arguments, never the arguments themselves.

What each entry is bound to:

- a call that is judged: the bound invocation the tool is called with, defaults included;
- a call refused before judgment (arguments that are not a JSON object, an unknown tool, a
  signature the gate cannot judge, arguments that do not bind): the arguments as proposed.

What is not on the log: a turn the gate refuses as a whole by raising (an item type it does
not handle, a call without an id, a duplicate id). It raises before anything is judged, so
it runs nothing and writes no entry. A hosted tool never reaches the loop at all.
"""

from __future__ import annotations

from typing import Any, Optional

from openai.types.chat import ChatCompletion
from openai.types.responses import Response

from recusal.audit import AuditLog

from ._gate import Policy, Tools, gate_function_calls, gate_tool_calls

#: The surface labels: which API shape's loop adjudicated the call.
SURFACE_RESPONSES = "openai.responses.function_call"
SURFACE_CHAT = "openai.chat_completions.tool_call"


def audited_function_calls(
    response: Response,
    tools: Tools,
    policy: Policy,
    log: AuditLog,
    *,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Responses API: ``gate_function_calls`` with every call recorded under its ``call_id``."""
    return gate_function_calls(
        response,
        tools,
        policy,
        audit=log,
        surface=SURFACE_RESPONSES,
        actor=actor,
        control=control,
    )


def audited_tool_calls(
    completion: ChatCompletion,
    tools: Tools,
    policy: Policy,
    log: AuditLog,
    *,
    choice: Optional[int] = None,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Chat Completions: ``gate_tool_calls`` with every call recorded under its ``id``."""
    return gate_tool_calls(
        completion,
        tools,
        policy,
        choice=choice,
        audit=log,
        surface=SURFACE_CHAT,
        actor=actor,
        control=control,
    )
