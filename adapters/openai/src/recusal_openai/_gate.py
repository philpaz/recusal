"""The gate for a loop the application owns: one decision function, two API shapes.

An application that calls the OpenAI API directly reads the function calls out of the
response, runs them, and sends the results back. That loop belongs to the application,
which makes it the place to put the gate:

- the Responses API: ``function_call`` output items, answered with ``function_call_output``
  input items (``gate_function_calls``);
- Chat Completions: ``tool_calls`` on the assistant message, answered with ``role: "tool"``
  messages (``gate_tool_calls``).

For every call the model proposes, the gate decodes the arguments the way the tool would
receive them, judges the call with ``recusal.decide``, runs the tool only if it is not
denied, and returns exactly one answer carrying that call's own id. Arguments that are not
valid JSON or not a JSON object, a tool the application does not have, and a policy that
raises are all refused with a reason, never run.

Default-deny on what the loop does not understand: an output item or tool call of a type
this gate does not handle (``custom_tool_call``, a hosted tool call, a type added in a later
SDK release) raises, naming the type, instead of being skipped. The SDK's answer types do
not require a call id, so the gate refuses a call without one and checks that every call
got exactly one answer.

Hosted tools (web search, file search, code interpreter, image generation, remote MCP) run
on the provider's side and never reach this loop, so nothing here can gate them.
"""

from __future__ import annotations

import copy
import inspect
import json
from typing import Any, Callable, List, Optional

from openai.types.chat import ChatCompletion
from openai.types.responses import Response

from recusal import Finding, decide
from recusal.invocation import unsupported_signature as _unsupported_signature

#: A Recusal policy: ``policy(tool_name, tool_input)`` returns a list of findings.
Policy = Callable[[str, dict], List[Any]]
#: The application's tools by the name the model calls them.
Tools = dict[str, Callable[..., Any]]

#: Response output items that cannot run anything, so the gate lets them pass. Everything
#: else (custom_tool_call, hosted tool calls, a type a later SDK adds) raises.
PASSIVE_OUTPUT_TYPES = frozenset({"message", "reasoning"})


def _unjudgeable(name: str, plain: str, detail: str, proposal: Any, record: dict) -> str:
    """The answer for a call the gate refuses before any policy runs.

    Without recording arguments it is ``plain``, the text this gate has always returned.
    With them (the audited loop) the refusal goes through ``decide`` like
    every other, so it is one entry on the log, bound to the call as the model proposed it,
    and the text the model reads is the reason on the record.
    """
    if not record:
        return plain
    bound = proposal if isinstance(proposal, dict) else {"input": proposal}

    def unjudgeable(_name, _input):
        return [Finding.fail("recusal_unjudgeable_input", severity="CRITICAL", message=detail)]

    try:
        _, reason = decide(name, bound, unjudgeable, safe_errors=True, **record)
    except Exception as exc:  # noqa: BLE001, the refusal stands even if the record cannot be written
        return f"Recusal failed closed (policy error): RECUSAL_POLICY_ERROR ({type(exc).__name__})"
    return reason


def judge_and_run(
    name: str, arguments: str | None, tools: Tools, policy: Policy, **record: Any
) -> str:
    """Decide one proposed call and run it if allowed; return the text the model gets back.

    The ONE decision path for both API shapes. The decoded arguments are bound to the
    tool's signature with its defaults applied (``inspect.signature(...).bind`` then
    ``apply_defaults``: argument binding, not type validation; no coercion happens on this
    path), the policy judges a detached copy of that bound invocation, and the tool is
    called with exactly those bound arguments, so a default the model omitted is judged at
    the value the function would use. A tool whose signature cannot be judged by name
    (variadic or positional-only parameters, an unsupported annotation, a default outside
    the JSON domain) is refused, as is a call that does not bind (an unknown or missing
    parameter; the refusal names the parameter, never a value).

    ``record`` goes straight to ``decide`` (``audit=``, ``surface=``, ``tool_use_id=``, ...),
    so a loop that puts the decision on a log (``audited_function_calls``) decides,
    and refuses, exactly as this one does. The entry for a judged call carries the
    fingerprint of the bound invocation the tool is called with, defaults included; the
    entry for a call refused before judgment is bound to the arguments as proposed.
    """
    try:
        args = json.loads(arguments or "{}")
    except ValueError:
        detail = "arguments are not valid JSON"
        return _unjudgeable(name, f"Recusal refused `{name}`: {detail}", detail, arguments, record)
    if not isinstance(args, dict):
        detail = "arguments are not a JSON object"
        return _unjudgeable(name, f"Recusal refused `{name}`: {detail}", detail, args, record)
    if name not in tools:
        return _unjudgeable(
            name, f"Recusal refused: unknown tool {name!r}", "unknown tool", args, record
        )
    func = tools[name]
    why = _unsupported_signature(func)
    if why is not None:
        detail = f"its signature cannot be judged ({why})"
        return _unjudgeable(name, f"Recusal refused `{name}`: {detail}", detail, args, record)
    try:
        bound = inspect.signature(func).bind(**args)
    except TypeError as exc:  # the message names parameters only, never values
        detail = f"arguments do not bind to the tool ({exc})"
        return _unjudgeable(name, f"Recusal refused `{name}`: {detail}", detail, args, record)
    bound.apply_defaults()
    invocation = dict(bound.arguments)
    try:
        decision, reason = decide(
            name, copy.deepcopy(invocation), policy, safe_errors=True, **record
        )
    except Exception as exc:  # noqa: BLE001, a buggy policy must not silently disable the gate
        return f"Recusal failed closed (policy error): RECUSAL_POLICY_ERROR ({type(exc).__name__})"
    if decision == "deny":
        return reason
    return str(func(**invocation))


def _require_id(call_id: Any, what: str) -> str:
    """The SDK does not require an id on the answer; the adapter does, or the answer is lost."""
    if not isinstance(call_id, str) or not call_id:
        raise ValueError(f"Recusal refused {what}: it has no call id to answer")
    return call_id


def _for_call(record: dict, call_id: str) -> dict:
    """The recording arguments for one call of a turn: the turn's, plus the call's own id."""
    return {**record, "tool_use_id": call_id} if record else {}


def _check_ids(call_ids: list[str]) -> None:
    """Refuse a turn with a duplicate id BEFORE any call in it runs: the turn's answers are
    keyed by id, so a duplicate could not be answered once, and a refusal raised after
    dispatch would leave a side effect behind for the caller's retry to repeat."""
    if len(set(call_ids)) != len(call_ids):
        raise ValueError(f"Recusal refused the turn: duplicate call ids {call_ids}")


def _check_answered_once(call_ids: list[str], answered: list[str]) -> None:
    """Every proposed call gets exactly one answer, and no answer invents a call."""
    if sorted(call_ids) != sorted(answered):
        raise ValueError(
            f"Recusal refused the turn: calls {sorted(call_ids)} but answers {sorted(answered)}"
        )
    if len(set(call_ids)) != len(call_ids):
        raise ValueError(f"Recusal refused the turn: duplicate call ids {call_ids}")


def gate_function_calls(
    response: Response, tools: Tools, policy: Policy, **record: Any
) -> list[dict[str, Any]]:
    """Responses API: judge every ``function_call`` item; return ``function_call_output`` items.

    Raises TypeError for an output item type this adapter does not handle, and ValueError for
    a call without an id. Pass the returned list as the next request's ``input``.
    ``record`` is the recording arguments for the turn (``audit=``, ``surface=``, ...);
    each call is recorded under its own ``call_id``. A turn refused as a whole raises
    before anything is judged, so it runs nothing and writes no entry.
    """
    # pass 1: every item is one this adapter handles and carries a usable, unique id;
    # nothing runs until the whole turn has passed, so a refused turn has run nothing
    calls: list[tuple[str, str, Any]] = []
    for item in response.output:
        item_type = getattr(item, "type", None)
        if item_type in PASSIVE_OUTPUT_TYPES:
            continue
        if item_type != "function_call":
            raise TypeError(
                f"Recusal refused output item of type {item_type!r}: "
                "this adapter handles function_call items only"
            )
        call_id = _require_id(getattr(item, "call_id", None), f"function_call {item.name!r}")
        calls.append((call_id, item.name, item.arguments))
    call_ids = [call_id for call_id, _, _ in calls]
    _check_ids(call_ids)
    # pass 2: judge and run each call, answering it under its own id
    outputs = [
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": judge_and_run(name, arguments, tools, policy, **_for_call(record, call_id)),
        }
        for call_id, name, arguments in calls
    ]
    _check_answered_once(call_ids, [o["call_id"] for o in outputs])
    return outputs


def _chosen(completion: ChatCompletion, choice: Any):
    """The one assistant message this turn dispatches: ``completion.choices[choice]``.

    A completion requested with ``n > 1`` carries alternative messages, not one batch, so
    one must be chosen explicitly; dispatching every alternative would run tool calls the
    application never selected. ``choice`` is the list position. With one choice it may be
    omitted. Every refusal here is raised before any call is judged or run.
    """
    choices = list(completion.choices or [])
    if not choices:
        raise ValueError("Recusal refused the completion: it has no choices")
    if choice is None:
        if len(choices) != 1:
            raise ValueError(
                f"Recusal refused the completion: it has {len(choices)} choices; pass "
                "choice=<list position> to dispatch one assistant message"
            )
        return choices[0]
    if isinstance(choice, bool) or not isinstance(choice, int):
        raise TypeError(
            f"Recusal refused the completion: choice must be an int list position, not "
            f"{type(choice).__name__}"
        )
    if choice < 0 or choice >= len(choices):
        raise ValueError(
            f"Recusal refused the completion: choice {choice} is out of range for "
            f"{len(choices)} choice(s)"
        )
    return choices[choice]


def gate_tool_calls(
    completion: ChatCompletion,
    tools: Tools,
    policy: Policy,
    *,
    choice: Optional[int] = None,
    **record: Any,
) -> list[dict[str, Any]]:
    """Chat Completions: judge every ``tool_calls`` entry of ONE chosen assistant message;
    return ``role: "tool"`` messages.

    ``choice`` is the list position in ``completion.choices``; it may be omitted only when
    the completion has exactly one choice. A completion with several choices (``n > 1``)
    and no ``choice`` is refused before anything is judged: the choices are alternative
    messages, and only the one the application selects is dispatched. Raises TypeError for
    a tool call type this adapter does not handle (``custom``) or a non-int ``choice``, and
    ValueError for a call without an id, no choices, or an out-of-range ``choice``. Append
    the returned messages after the chosen assistant message in the next request.
    ``record`` is the recording arguments for the turn, as in ``gate_function_calls``.
    """
    chosen = _chosen(completion, choice)
    calls: list[tuple[str, str, Any]] = []
    for call in chosen.message.tool_calls or []:
        if call.type != "function":
            raise TypeError(
                f"Recusal refused tool call of type {call.type!r}: "
                "this adapter handles function tool calls only"
            )
        call_id = _require_id(call.id, f"tool call {call.function.name!r}")
        calls.append((call_id, call.function.name, call.function.arguments))
    call_ids = [call_id for call_id, _, _ in calls]
    _check_ids(call_ids)
    outputs = [
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": judge_and_run(name, arguments, tools, policy, **_for_call(record, call_id)),
        }
        for call_id, name, arguments in calls
    ]
    _check_answered_once(call_ids, [o["tool_call_id"] for o in outputs])
    return outputs
