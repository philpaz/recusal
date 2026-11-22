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
a string, not valid JSON (``NaN`` and ``Infinity`` are not JSON), not a JSON object, that
hold a number Python reads as infinity or are nested more than 64 levels deep, a tool the
application does not have, and a policy that raises are all refused with a reason, never
run. An empty string is a call with no arguments.

A call that is not finished is refused the same way, before the policy runs: a
``function_call`` item whose ``status`` is not ``completed`` (an item with no status
passes), every function call of a response whose ``status`` is not ``completed`` (a
response with no status passes), and every tool call of a Chat Completions choice whose
``finish_reason`` is not ``stop``, ``tool_calls`` or ``function_call``. So is a
``function_call`` item that carries a ``namespace``: the tools are keyed by bare name, so
the gate could not tell two functions of the same name apart.

The gate calls tools synchronously. An ``async def`` tool is refused with a reason before
the policy runs, because calling it would only create a coroutine that nothing awaits. A
generator function is refused for the same reason: calling it runs none of its body.

A tool that raises: by default the exception leaves the gate, as it always has. Calls
earlier in the turn have run and their answers are lost with it, and calls later in the
turn do not run. ``answer_tool_errors=True`` answers the failed call with its exception's
class name under its own call id instead, and the rest of the turn is judged and run as
usual; the traceback goes to the ``recusal_openai`` logger, never to the model. The class
named is the exception's nearest built-in one, so no name an application chose is sent.

Default-deny on what the loop does not understand: an output item or tool call of a type
this gate does not handle (``custom_tool_call``, a hosted tool call, a type added in a later
SDK release) raises, naming the type, instead of being skipped. So does a Chat Completions
message that carries the legacy ``function_call`` field, which this gate does not read.
The SDK's answer types do not require a call id, so the gate refuses a call without one
and checks that every call got exactly one answer. A keyword argument the gate does not
know raises ``TypeError`` before anything is judged.

The gate keeps no state between turns: a call id is checked within one turn only.

Hosted tools (web search, file search, code interpreter, image generation, remote MCP) run
on the provider's side, so nothing here can gate them. What reaches this loop is the
provider's record of the call (``web_search_call``, ``mcp_call``, ...). By default such a
record raises like any other unhandled item; ``allow_hosted_records=True`` lets a finished
record of a type in ``HOSTED_RECORD_TYPES`` pass, unjudged and unrecorded. A remote MCP
server configured to ask for approval sends an ``mcp_approval_request`` item, which this
gate does not judge: it raises, so the call is never approved by this loop.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import math
from typing import Any, Callable, List, Optional

from openai.types.chat import ChatCompletion
from openai.types.responses import Response

from recusal import Finding, decide
from recusal.invocation import detached as _detached
from recusal.invocation import unsupported_signature as _unsupported_signature

#: A Recusal policy: ``policy(tool_name, tool_input)`` returns a list of findings.
Policy = Callable[[str, dict], List[Any]]
#: The application's tools by the name the model calls them.
Tools = dict[str, Callable[..., Any]]

#: Response output items that cannot run anything, so the gate lets them pass. A
#: ``function_call`` is judged. Everything else raises (custom_tool_call, a type a later SDK
#: adds, and the provider's own records unless ``allow_hosted_records=True`` admits them).
PASSIVE_OUTPUT_TYPES = frozenset({"message", "reasoning"})

#: Items the provider produced on its own side: the record of a hosted tool call it ran,
#: and ``compaction``, its summary of earlier context. Nothing in the loop can stop or
#: judge them. They raise by default; ``gate_function_calls(..., allow_hosted_records=True)``
#: lets exactly these pass, and only when finished (``HOSTED_RECORD_FINISHED``).
#: ``mcp_approval_request`` is not here: it asks the loop for a decision, and this gate
#: does not make that one.
HOSTED_RECORD_TYPES = frozenset(
    {
        "web_search_call",
        "file_search_call",
        "code_interpreter_call",
        "image_generation_call",
        "mcp_call",
        "mcp_list_tools",
        "compaction",
    }
)

#: The ``status`` values of a hosted record that is over. ``None`` is an item that carries
#: no status (``compaction`` and ``mcp_list_tools`` never do; an ``mcp_call`` may not). A
#: record still ``in_progress`` (or ``searching``, ``calling``, ...) is not a finished
#: record and raises even with ``allow_hosted_records=True``.
HOSTED_RECORD_FINISHED = frozenset({None, "completed", "failed", "incomplete"})

#: The ``status`` values of a ``function_call`` item, and of a response, that is finished.
#: ``None`` is one that carries no status. Anything else (``in_progress``, ``incomplete``,
#: ``failed``, ``cancelled``, ``queued``, a value a later SDK adds) is not finished, and a
#: function call that is not finished is refused: its arguments may have been cut off.
_FUNCTION_CALL_FINISHED = frozenset({None, "completed"})

#: The ``finish_reason`` values of a Chat Completions choice whose tool calls are finished.
#: ``length`` (cut off at the token limit), ``content_filter`` (content omitted) and
#: anything else, a missing reason included, refuse every tool call of the choice.
_CHAT_FINISHED = frozenset({"stop", "tool_calls", "function_call"})

#: The keywords the gate passes on to ``recusal.decide``. Any other keyword raises
#: ``TypeError`` before anything is judged, so a mistyped one is never swallowed.
_RECORD_KEYWORDS = frozenset(
    {"audit", "surface", "tool_use_id", "actor", "control", "allow_on_pass", "fail_closed"}
)

#: How many levels below the arguments object a value may sit. The kernel's ``json_data``
#: draws the same line for a default from 0.15.0; the gate holds its own so that it
#: refuses the same calls on every kernel release it supports.
_MAX_DEPTH = 64

_TOO_DEEP = "arguments are nested too deeply"

_LOG = logging.getLogger("recusal_openai")


def _check_record(record: dict) -> None:
    """Raise ``TypeError`` for a keyword that is neither the gate's own nor ``decide``'s."""
    unknown = sorted(set(record) - _RECORD_KEYWORDS)
    if unknown:
        raise TypeError(
            f"Recusal refused the turn: unexpected keyword argument {unknown[0]!r}; the "
            f"keywords passed on to recusal.decide are {', '.join(sorted(_RECORD_KEYWORDS))}"
        )


class _NotFinite(ValueError):
    """A JSON number Python reads as infinity (``1e999``)."""


def _finite(token: str) -> float:
    """``json.loads``'s ``parse_float``: the float, or ``_NotFinite`` for an infinite one."""
    number = float(token)
    if not math.isfinite(number):
        raise _NotFinite
    return number


def _too_deep(value: Any, depth: int = 0) -> bool:
    """Whether decoded arguments hold a value more than ``_MAX_DEPTH`` levels down."""
    if depth > _MAX_DEPTH:
        return True
    if type(value) is dict:
        return any(_too_deep(item, depth + 1) for item in value.values())
    if type(value) is list:
        return any(_too_deep(item, depth + 1) for item in value)
    return False


def _not_json(token: str) -> Any:
    """``json.loads``'s ``parse_constant``: ``NaN``, ``Infinity`` and ``-Infinity`` are
    tokens Python accepts and JSON does not have."""
    raise ValueError(f"{token} is not JSON")


def _is_async(func: Callable[..., Any]) -> bool:
    """Whether ``func`` is declared async, so calling it would run nothing.

    This reads the declaration (``async def``, or a function marked with
    ``inspect.markcoroutinefunction``) through a partial, a staticmethod, a bound method
    and a callable object. A wrapper that hides the declaration is not seen here; what it
    returns is (``_answer``).
    """
    target: Any = func
    while isinstance(target, (functools.partial, staticmethod)):
        target = target.func if isinstance(target, functools.partial) else target.__func__
    if inspect.iscoroutinefunction(target) or inspect.isasyncgenfunction(target):
        return True
    call = getattr(type(target), "__call__", None)
    return inspect.iscoroutinefunction(call) or inspect.isasyncgenfunction(call)


def _is_generator(func: Callable[..., Any]) -> bool:
    """Whether ``func`` is declared as a generator function, so calling it would run nothing.

    Read the way ``_is_async`` reads ``async def``. A wrapper that hides the declaration is
    not seen here; what it returns is (``_answer``).
    """
    target: Any = func
    while isinstance(target, (functools.partial, staticmethod)):
        target = target.func if isinstance(target, functools.partial) else target.__func__
    call = getattr(type(target), "__call__", None)
    return inspect.isgeneratorfunction(target) or inspect.isgeneratorfunction(call)


def _answer(name: str, result: Any) -> str:
    """The text for what an allowed tool returned.

    A coroutine or an async generator is not a result: the gate did not run it, and says
    so instead of answering with the object's ``repr``. One that was never started is
    closed, which runs none of its code and only spares the "never awaited" warning. One
    that has started belongs to whoever is running it and is left alone. A generator is
    not a result either: none of its body has run, and it is answered as not run and left
    alone. Any other value, a future or another awaitable included, is answered as its
    ``str``, as before.
    """
    if inspect.iscoroutine(result) or inspect.isasyncgen(result):
        if inspect.iscoroutine(result) and inspect.getcoroutinestate(result) == "CORO_CREATED":
            result.close()
        return (
            f"Tool `{name}` returned a coroutine, not a result: "
            "this gate calls tools synchronously and did not run it"
        )
    if inspect.isgenerator(result):
        return f"Tool `{name}` returned a generator, not a result: this gate did not run it"
    return str(result)


def _unbound(parameters: Any, args: dict) -> str:
    """Why ``args`` do not bind to a tool with ``parameters``, in the tool's own words.

    ``inspect``'s own message quotes an unexpected argument's key, which the model chose.
    This names only parameters the tool declares, and counts the arguments it does not.
    """
    missing = [
        f"`{name}`"
        for name, parameter in parameters.items()
        if name not in args and parameter.default is parameter.empty
    ]
    extra = sum(1 for key in args if key not in parameters)
    parts = []
    if missing:
        parts.append(f"missing required {', '.join(missing)}")
    if extra:
        parts.append(f"{extra} argument{'s' if extra != 1 else ''} the tool does not take")
    return f"arguments do not bind to the tool ({'; '.join(parts)})"


def _raised(exc: Exception) -> str:
    """The name the model reads for a tool's exception: its nearest built-in class.

    A class defined by an application or a library is named by text someone chose, which
    could carry anything to the model. The built-in it derives from says what kind of
    failure it was from a fixed vocabulary (``TimeoutError``, ``ValueError``, ...,
    ``Exception`` at worst).
    """
    for cls in type(exc).__mro__:
        if cls.__module__ == "builtins":
            return cls.__name__
    return "Exception"


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
    name: str,
    arguments: str,
    tools: Tools,
    policy: Policy,
    *,
    answer_tool_errors: bool = False,
    **record: Any,
) -> str:
    """Decide one proposed call and run it if allowed; return the text the model gets back.

    The ONE decision path for both API shapes. The decoded arguments are bound to the
    tool's signature with its defaults applied (``inspect.signature(...).bind`` then
    ``apply_defaults``: argument binding, not type validation; no coercion happens on this
    path), the policy judges a detached copy of that bound invocation, and the tool is
    called with exactly those bound arguments, so a default the model omitted is judged at
    the value the function would use. A tool whose signature cannot be judged by name
    (variadic or positional-only parameters, an unsupported annotation, a default outside
    the JSON domain) is refused, as is a call that does not bind (the refusal names a
    missing required parameter and counts the arguments the tool does not take; it never
    quotes a key or a value the model sent).

    ``arguments`` is the JSON text of the call. Only a string is parsed, and an empty
    string is a call with no arguments; anything else is refused. So are ``NaN``,
    ``Infinity`` and ``-Infinity`` (not JSON), a number Python reads as infinity
    (``1e999``), and arguments nested more than 64 levels below the arguments object or
    too deeply to parse. The copy the policy judges is made with
    ``recusal.invocation.detached``, so a policy that writes to it cannot change what the
    tool receives.

    The tool is called synchronously. An async tool and a generator function are refused
    before the policy runs, and a coroutine or a generator returned by an ordinary
    function is answered as not run.

    A tool that raises propagates by default. With ``answer_tool_errors=True`` it is answered with
    ``Tool `name` raised <ExceptionClass>``, where the class is the exception's nearest
    built-in one: never its message, and never a name an application chose. The traceback
    goes to the ``recusal_openai`` logger; a logging handler that fails does not change
    the answer. Only ``Exception`` is caught, so a keyboard interrupt still propagates.

    ``record`` goes straight to ``decide`` (``audit=``, ``surface=``, ``tool_use_id=``, ...),
    so a loop that puts the decision on a log (``audited_function_calls``) decides,
    and refuses, exactly as this one does. The entry for a judged call carries the
    fingerprint of the bound invocation the tool is called with, defaults included; the
    entry for a call refused before judgment is bound to the arguments as proposed: the
    decoded object, or the text as sent when it is the text that was refused.

    A keyword in ``record`` that ``decide`` does not take raises ``TypeError``.
    """
    _check_record(record)
    if not isinstance(arguments, str):
        detail = "arguments are not a string"
        return _unjudgeable(name, f"Recusal refused `{name}`: {detail}", detail, arguments, record)
    try:
        args = json.loads(arguments or "{}", parse_constant=_not_json, parse_float=_finite)
    except _NotFinite:
        detail = "a number in the arguments is not finite"
        return _unjudgeable(name, f"Recusal refused `{name}`: {detail}", detail, arguments, record)
    except RecursionError:
        detail = _TOO_DEEP
        return _unjudgeable(name, f"Recusal refused `{name}`: {detail}", detail, arguments, record)
    except ValueError:
        detail = "arguments are not valid JSON"
        return _unjudgeable(name, f"Recusal refused `{name}`: {detail}", detail, arguments, record)
    # nothing nested deeper is fingerprinted, copied or handed to a policy, each of which
    # walks the arguments and would fail at a depth that depends on the caller's stack
    if _too_deep(args):
        detail = _TOO_DEEP
        return _unjudgeable(name, f"Recusal refused `{name}`: {detail}", detail, arguments, record)
    if not isinstance(args, dict):
        detail = "arguments are not a JSON object"
        return _unjudgeable(name, f"Recusal refused `{name}`: {detail}", detail, args, record)
    if name not in tools:
        return _unjudgeable(
            name, f"Recusal refused: unknown tool {name!r}", "unknown tool", args, record
        )
    func = tools[name]
    if _is_async(func):
        detail = "it is declared async, and this gate calls tools synchronously"
        return _unjudgeable(name, f"Recusal refused `{name}`: {detail}", detail, args, record)
    if _is_generator(func):
        detail = "it is a generator function, and calling it would run none of its body"
        return _unjudgeable(name, f"Recusal refused `{name}`: {detail}", detail, args, record)
    why = _unsupported_signature(func)
    if why is not None:
        detail = f"its signature cannot be judged ({why})"
        return _unjudgeable(name, f"Recusal refused `{name}`: {detail}", detail, args, record)
    signature = inspect.signature(func)
    try:
        bound = signature.bind(**args)
    except TypeError:  # its text can quote a key the model chose, so it is not used
        detail = _unbound(signature.parameters, args)
        return _unjudgeable(name, f"Recusal refused `{name}`: {detail}", detail, args, record)
    bound.apply_defaults()
    invocation = dict(bound.arguments)
    try:
        judged = _detached(invocation)
    except (TypeError, RecursionError):  # the policy has not run: a refusal, on the record
        detail = "the invocation cannot be copied as JSON data for the policy"
        return _unjudgeable(name, f"Recusal refused `{name}`: {detail}", detail, args, record)
    try:
        decision, reason = decide(name, judged, policy, safe_errors=True, **record)
    except Exception as exc:  # noqa: BLE001, a buggy policy must not silently disable the gate
        return f"Recusal failed closed (policy error): RECUSAL_POLICY_ERROR ({type(exc).__name__})"
    if decision == "deny":
        return reason
    if not answer_tool_errors:
        return _answer(name, func(**invocation))
    try:
        return _answer(name, func(**invocation))
    except Exception as exc:  # noqa: BLE001, the caller asked for a failed tool to be answered
        try:
            _LOG.exception("tool %r raised after the gate allowed the call", name)
        except Exception:  # noqa: BLE001, a broken log handler must not unanswer the call
            pass
        return f"Tool `{name}` raised {_raised(exc)}"


def _answer_call(
    name: str,
    arguments: Any,
    refused: Optional[str],
    tools: Tools,
    policy: Policy,
    answer_tool_errors: bool,
    record: dict,
) -> str:
    """The answer for one call of a turn: the refusal the loop found while reading it
    (``refused``: a namespace, a call that is not finished), or ``judge_and_run``.

    A refusal found by the loop is made before the arguments are read, so on the record it
    is bound to them as sent.
    """
    if refused is not None:
        return _unjudgeable(
            name, f"Recusal refused `{name}`: {refused}", refused, arguments, record
        )
    return judge_and_run(
        name, arguments, tools, policy, answer_tool_errors=answer_tool_errors, **record
    )


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
    response: Response,
    tools: Tools,
    policy: Policy,
    *,
    allow_hosted_records: bool = False,
    answer_tool_errors: bool = False,
    **record: Any,
) -> list[dict[str, Any]]:
    """Responses API: judge every ``function_call`` item; return ``function_call_output`` items.

    Raises TypeError for an output item type this adapter does not handle or a keyword it
    does not know, and ValueError for a call without an id. Pass the returned list as the
    next request's ``input``. A ``function_call`` item that carries a ``namespace``, one
    whose ``status`` is not ``completed`` (no status passes), and every function call of
    a response whose ``status`` is not ``completed`` (no status passes) are refused with a
    reason under their own ids, before the policy runs.
    ``allow_hosted_records=True`` lets a finished item of a type in ``HOSTED_RECORD_TYPES``
    pass: the provider produced it on its own side, this loop can neither judge nor stop
    it, so it is not answered and not put on the record. ``answer_tool_errors=True``
    answers a tool that raises instead of letting the exception leave the gate; see
    ``judge_and_run``.
    ``record`` is the recording arguments for the turn (``audit=``, ``surface=``, ...);
    each call is recorded under its own ``call_id``. A turn refused as a whole raises
    before anything is judged, so it runs nothing and writes no entry.
    """
    _check_record(record)
    unfinished = getattr(response, "status", None) not in _FUNCTION_CALL_FINISHED
    # pass 1: every item is one this adapter handles and carries a usable, unique id;
    # nothing runs until the whole turn has passed, so a refused turn has run nothing
    calls: list[tuple[str, str, Any, Optional[str]]] = []
    for item in response.output:
        item_type = getattr(item, "type", None)
        if item_type in PASSIVE_OUTPUT_TYPES:
            continue
        if item_type in HOSTED_RECORD_TYPES:
            status = getattr(item, "status", None)
            if not allow_hosted_records:
                raise TypeError(
                    f"Recusal refused output item of type {item_type!r}: the provider "
                    "produced it on its own side, where this gate cannot judge it; pass "
                    "allow_hosted_records=True to let finished records through"
                )
            if status not in HOSTED_RECORD_FINISHED:
                raise TypeError(
                    f"Recusal refused output item of type {item_type!r}: its status is "
                    f"{status!r}, so it is not a finished record"
                )
            continue
        if item_type != "function_call":
            raise TypeError(
                f"Recusal refused output item of type {item_type!r}: "
                "this adapter handles function_call items only"
            )
        # the type check above is on the item's ``type`` string, which a type checker cannot
        # use to narrow the SDK's union of output items; from here it is a function call
        call: Any = item
        call_id = _require_id(getattr(call, "call_id", None), f"function_call {call.name!r}")
        refused = None
        if getattr(call, "namespace", None):
            # the tools and the policy know a function by its bare name only
            refused = "it carries a namespace, and namespaced function calls are not supported"
        elif unfinished:
            refused = "the response is not finished (its status is not `completed`)"
        elif getattr(call, "status", None) not in _FUNCTION_CALL_FINISHED:
            refused = "the call is not finished (its status is not `completed`)"
        calls.append((call_id, call.name, call.arguments, refused))
    call_ids = [call_id for call_id, _, _, _ in calls]
    _check_ids(call_ids)
    # pass 2: judge and run each call, answering it under its own id
    outputs = [
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": _answer_call(
                name,
                arguments,
                refused,
                tools,
                policy,
                answer_tool_errors,
                _for_call(record, call_id),
            ),
        }
        for call_id, name, arguments, refused in calls
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
    answer_tool_errors: bool = False,
    **record: Any,
) -> list[dict[str, Any]]:
    """Chat Completions: judge every ``tool_calls`` entry of ONE chosen assistant message;
    return ``role: "tool"`` messages.

    ``choice`` is the list position in ``completion.choices``; it may be omitted only when
    the completion has exactly one choice. A completion with several choices (``n > 1``)
    and no ``choice`` is refused before anything is judged: the choices are alternative
    messages, and only the one the application selects is dispatched. Raises TypeError for
    a tool call type this adapter does not handle (``custom``), a chosen message that
    carries the legacy ``function_call`` field, a keyword it does not know or a non-int
    ``choice``, and ValueError for a call without an id, no choices, or an out-of-range
    ``choice``. Append the returned messages after the chosen assistant message in the
    next request. When the chosen choice's ``finish_reason`` is not ``stop``,
    ``tool_calls`` or ``function_call`` (``length``: the message was cut off), every tool
    call in it is refused with a reason under its own id, before the policy runs.
    ``record`` is the recording arguments for the turn, and ``answer_tool_errors`` the
    same keyword, as in ``gate_function_calls``.
    """
    _check_record(record)
    chosen = _chosen(completion, choice)
    if getattr(chosen.message, "function_call", None):
        raise TypeError(
            "Recusal refused the message: it carries the legacy `function_call` field; "
            "this adapter handles `tool_calls` only"
        )
    refused = None
    if getattr(chosen, "finish_reason", None) not in _CHAT_FINISHED:
        refused = (
            "the message is not finished (its finish_reason is not `stop`, `tool_calls` "
            "or `function_call`)"
        )
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
            "content": _answer_call(
                name,
                arguments,
                refused,
                tools,
                policy,
                answer_tool_errors,
                _for_call(record, call_id),
            ),
        }
        for call_id, name, arguments in calls
    ]
    _check_answered_once(call_ids, [o["tool_call_id"] for o in outputs])
    return outputs
