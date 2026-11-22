"""The audited loop: the same decision, with every call on a hash-chained ``AuditLog``.

``audited_function_calls`` and ``audited_tool_calls`` call ``gate_function_calls`` and
``gate_tool_calls`` with the recording arguments passed through to ``recusal.decide``,
which writes the entry before returning. There is no second decision path. If the log
cannot be written the decision becomes a deny, so nothing runs unrecorded.

Each entry carries the surface of the API shape that adjudicated the call and the API's own
call id, and a SHA-256 fingerprint of the arguments, never the arguments themselves.

What each entry is bound to:

- a call that is judged: the bound invocation the tool is called with, defaults included;
- a call refused before judgment for what its decoded arguments are or name (not a JSON
  object, an unknown tool, an async or generator tool, a signature the gate cannot judge,
  arguments that do not bind, a list or an object for a tool cached with
  ``functools.lru_cache``): the decoded arguments as proposed;
- a call refused before its arguments are decoded, or because they cannot be (a call
  that carries a namespace, a call that is not finished, arguments that are not a string,
  not valid JSON, hold a number that is not finite or are nested too deeply): the
  arguments as sent, as ``{"input": <arguments>}``.

A refusal's reason on the log names parameters the tool declares. It never quotes an
argument's key or value. A tool the application has is on the entry under its name. A
name it has no tool for is the model's text: it is on the entry, and in the reason, as
sent only if it is 1 to 64 printable characters with no backtick, and otherwise as a
description of its length.

What is not on the log: a turn the gate refuses as a whole by raising (an item type it does
not handle, a Chat Completions message with the legacy ``function_call`` field, a call
without an id, a duplicate id, a keyword the gate does not take). It raises before
anything is judged, so
it runs nothing and writes no entry. A hosted tool runs on the provider's side: its record
is never an entry, whether it raises or passes with ``allow_hosted_records=True``.

An entry records the gate's decision, not what the tool then did. A call the gate allowed
whose tool raised, returned a coroutine nothing ran, or returned an iterator the gate did
not consume, is on the log as allowed.
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
    allow_hosted_records: bool = False,
    answer_tool_errors: bool = False,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Responses API: ``gate_function_calls`` with every call recorded under its ``call_id``."""
    return gate_function_calls(
        response,
        tools,
        policy,
        allow_hosted_records=allow_hosted_records,
        answer_tool_errors=answer_tool_errors,
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
    answer_tool_errors: bool = False,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Chat Completions: ``gate_tool_calls`` with every call recorded under its ``id``."""
    return gate_tool_calls(
        completion,
        tools,
        policy,
        choice=choice,
        answer_tool_errors=answer_tool_errors,
        audit=log,
        surface=SURFACE_CHAT,
        actor=actor,
        control=control,
    )
