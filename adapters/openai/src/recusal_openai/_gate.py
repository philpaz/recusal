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
the gate could not tell two functions of the same name apart. The other fields of a
``function_call`` item are not read: a call the item marks ``async`` and a call whose
``caller`` is a program are judged and run like any other.

The gate calls tools synchronously. An ``async def`` tool is refused with a reason before
the policy runs, because calling it would only create a coroutine that nothing awaits. A
generator function is refused for the same reason: calling it runs none of its body. Both
are read through a ``functools.partial``, a ``staticmethod``, a bound method and
``functools.lru_cache``; a wrapper the application wrote (``functools.wraps`` included) is
not looked through, because it may await or consume what it wraps. A tool cached with
``functools.lru_cache`` is refused, before the policy runs, for a call that hands it a
list or an object: the cache cannot hash one.

Whether a tool's signature can be judged is the kernel's answer
(``recusal.invocation.unsupported_signature``). A reason it gives, and any exception it
raises, is a refusal of that one call before the policy runs.

A refusal names a tool the application has by its name. Any other name is the model's
text: one of 1 to 64 printable characters with no backtick is used as sent, and any other
(longer, empty, not a string, or holding a backtick or a character that does not print)
is described by its length and never quoted.

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
and checks that every call got exactly one answer. A call id is echoed in its answer and
written on its audit entry exactly as sent, so the gate takes one of at most 256
characters: a longer one raises before anything in the turn runs. A message the gate
raises quotes a call id only if it is printable and holds no backtick, and otherwise
describes it by its length.

A keyword argument the gate does not take raises ``TypeError`` before anything is judged,
as does ``audit=`` without ``surface=``, and ``surface=``, ``actor=`` or ``control=`` with
a value when ``audit=`` is not passed at all. ``audit=None`` passed explicitly is the
plain gate: those keywords are then accepted and not used. A ``surface=`` or ``control=``
that ``recusal.decide`` would refuse raises ``ValueError`` naming the keyword, before
anything is judged.

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
from collections.abc import AsyncIterator, Iterator
from typing import Any, Callable, List, Optional

from openai.types.chat import ChatCompletion
from openai.types.responses import Response

import recusal.invocation as _invocation
from recusal import Finding, decide
from recusal.invocation import detached as _detached
from recusal.invocation import json_data as _json_data
from recusal.invocation import unsupported_signature as _unsupported_signature

# ``reached_parameters`` is the kernel's walk of every callable a call reaches, new in
# recusal 0.15.0, and ``unsupported_signature`` refuses a decorated tool through it. With
# an older kernel that check is absent and such a tool would run with values no policy
# judged, so the package does not import there. The declared dependency says the same;
# this holds where an installer was told to ignore it.
if not hasattr(_invocation, "reached_parameters"):
    raise ImportError(
        "recusal-openai requires recusal>=0.15.0: an older kernel does not check the "
        "wrappers of a decorated tool"
    )

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
_RECORD_KEYWORDS = frozenset({"audit", "surface", "actor", "control", "allow_on_pass"})

#: The keywords ``judge_and_run`` takes: the loops', and the id of the one call it judges.
_CALL_KEYWORDS = _RECORD_KEYWORDS | {"tool_use_id"}

#: The keywords that mean something only on an audit log. Given a value when ``audit=``
#: is not passed at all they raise, so a caller who believes it is recording is never not
#: recording. With ``audit=None`` passed explicitly they are accepted and not used.
_NEED_AUDIT = ("surface", "tool_use_id", "actor", "control")

#: The recording keywords ``recusal.decide`` holds to a rule of its own (a surface that is
#: a nonempty name and not a reserved one, a control that is a dict, a call id that is a
#: nonempty string). ``_recording`` asks ``decide`` about each one, once, before any call
#: is looked at.
_CHECKED_BY_DECIDE = ("surface", "control", "tool_use_id")

#: A surface ``decide`` accepts, passed while it checks a keyword other than ``surface``.
_CHECK_SURFACE = "openai.keyword_check"

#: Told apart from ``audit=None``: the keyword was not passed at all.
_ABSENT: Any = object()

#: Keywords of ``recusal.decide`` a caller might expect here, and why each is not taken.
_NOT_TAKEN = {
    "tool_use_id": "the gate records each call under the call's own id",
    "fail_closed": "this gate always fails closed",
}

#: The longest tool name a refusal carries when the application has no tool of that name.
_NAME_SHOWN = 64

#: The longest call id the gate takes. The id is echoed in the answer and written on the
#: audit entry exactly as sent, since both are how the call is found again, so it cannot
#: be shortened there: a longer one refuses the turn before anything runs.
_ID_MAX = 256

#: What an audit entry is bound to when a call is refused before decoding and its
#: arguments, on an item the SDK did not validate, are not JSON data and so cannot be
#: fingerprinted.
_NOT_JSON_DATA = {"input": "<arguments that are not JSON data>"}

#: ``functools.lru_cache`` and ``functools.cache``: a wrapper that passes the call on to
#: the function it wraps, and hashes every argument to do so.
_LRU_CACHE = type(functools.lru_cache(maxsize=None)(lambda: None))

#: How many levels below the arguments object a value may sit. The kernel's ``json_data``
#: draws the same line for a default; the gate holds its own so that it refuses the same
#: calls whatever a later kernel release does with that line.
_MAX_DEPTH = 64

_TOO_DEEP = "arguments are nested too deeply"

_LOG = logging.getLogger("recusal_openai")


class _NoLog:
    """Stands where the audit log would be while ``decide`` checks a recording value:
    whatever is appended to it is dropped."""

    def append(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _no_findings(_name: str, _input: dict) -> List[Any]:
    """The policy ``decide`` is given while it checks a recording value."""
    return []


def _check_values(record: dict) -> None:
    """Raise ``ValueError``, naming the keyword, for a recording value ``decide`` refuses.

    The rule for each value is ``decide``'s own and is not restated here: ``decide`` is
    called once per keyword with a log that drops the entry, and a keyword it raises for
    is the one named. Left to the calls of the turn, ``decide`` would refuse every one of
    them, the gate could only report that as an error of a policy that never ran, and no
    entry would be written.

    A call id is also held to the gate's own bound (``_ID_MAX``).
    """
    for keyword in _CHECKED_BY_DECIDE:
        value = record.get(keyword)
        if value is None:
            continue
        asked: dict[str, Any] = {"audit": _NoLog(), "surface": _CHECK_SURFACE, keyword: value}
        try:
            decide("recusal_openai", {}, _no_findings, **asked)
        except RecursionError:  # the stack ran out, which says nothing about the value
            raise
        except Exception as exc:  # noqa: BLE001, whatever decide raises, the value is refused
            raise ValueError(
                f"Recusal refused the turn: {keyword}= is not a value recusal.decide "
                f"records ({exc})"
            ) from None
    call_id = record.get("tool_use_id")
    if isinstance(call_id, str) and len(call_id) > _ID_MAX:
        raise ValueError(
            f"Recusal refused the turn: tool_use_id= is {len(call_id)} characters long; "
            f"the gate takes a call id of at most {_ID_MAX}"
        )


def _recording(record: dict, taken: frozenset = _RECORD_KEYWORDS) -> dict:
    """The recording arguments to pass on to ``decide``: ``record`` with an audit log,
    nothing without one.

    ``audit=None`` passed explicitly is the plain gate, written so that a loop whose log
    is optional is one call (``audit=maybe_log, surface=...``): the other recording
    keywords are accepted and not used (their values are still checked). When ``audit=``
    is not passed at all, a keyword that needs it and has a value raises, so a caller who
    believes it is recording is never silently not recording.

    Raises ``TypeError`` for a keyword that is not in ``taken``, for ``audit=`` without
    ``surface=`` and for a keyword that needs ``audit=`` and has a value when ``audit=``
    is absent, and ``ValueError`` for a value ``decide`` would refuse (``_check_values``).
    ``decide`` would refuse each of these on every call, and the gate could only report
    that as an error of a policy that never ran.
    """
    unknown = sorted(set(record) - taken)
    if unknown:
        why = _NOT_TAKEN.get(unknown[0])
        raise TypeError(
            f"Recusal refused the turn: unexpected keyword argument {unknown[0]!r}"
            f"{f' ({why})' if why else ''}; the "
            f"keywords passed on to recusal.decide are {', '.join(sorted(_RECORD_KEYWORDS))}"
        )
    audit = record.get("audit", _ABSENT)
    if audit is _ABSENT:
        needs = [keyword for keyword in _NEED_AUDIT if record.get(keyword) is not None]
        if needs:
            raise TypeError(
                f"Recusal refused the turn: {needs[0]}= requires audit=; without an audit "
                "log the gate records nothing (pass audit=None for a log that is optional)"
            )
        return {}
    if audit is not None and record.get("surface") is None:
        raise TypeError("Recusal refused the turn: audit= requires surface=")
    _check_values(record)
    return record if audit is not None else {}


def _shown(name: Any, tools: Tools) -> str:
    """The name a refusal, and its audit entry, carry for the tool a call asked for.

    A name the application has a tool for is the application's own and is used as it is.
    Any other name is the model's text. One of 1 to ``_NAME_SHOWN`` printable characters
    with no backtick is used as sent. Any other (longer, empty, not a string, or holding
    a backtick or a character that does not print, such as a newline, a control or
    formatting character or a lone surrogate) is described by its length and never quoted:
    a refusal sets the name between backticks, and what a model can put there must not be
    able to close them, forge a line or grow the record without bound.
    """
    if not isinstance(name, str):
        return "<a tool name that is not a string>"
    if name in tools:
        return name
    if 0 < len(name) <= _NAME_SHOWN and name.isprintable() and "`" not in name:
        return name
    return f"<a tool name of {len(name)} characters, not shown>"


class _NotFinite(ValueError):
    """A JSON number Python reads as infinity (``1e999``)."""


def _finite(token: str) -> float:
    """``json.loads``'s ``parse_float``: the float, or ``_NotFinite`` for an infinite one."""
    number = float(token)
    if not math.isfinite(number):
        raise _NotFinite
    return number


def _too_deep(value: Any) -> bool:
    """Whether decoded arguments hold a value more than ``_MAX_DEPTH`` levels down.

    Walked with a list, not by recursion, so the answer does not depend on how much of
    the stack the caller has already used.
    """
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > _MAX_DEPTH:
            return True
        if type(item) is dict:
            pending.extend((inner, depth + 1) for inner in item.values())
        elif type(item) is list:
            pending.extend((inner, depth + 1) for inner in item)
    return False


def _not_json(token: str) -> Any:
    """``json.loads``'s ``parse_constant``: ``NaN``, ``Infinity`` and ``-Infinity`` are
    tokens Python accepts and JSON does not have."""
    raise ValueError(f"{token} is not JSON")


def _declared(func: Any) -> Any:
    """The function a call to ``func`` runs, read through the wrappers that only pass the
    call on: a ``functools.partial``, a ``staticmethod``, a bound method and
    ``functools.lru_cache``.

    A wrapper an application wrote is not looked through, ``functools.wraps`` or not: it
    may await or consume what it wraps, so how the wrapped function is declared does not
    say what a call to the wrapper runs.
    """
    target = func
    for _ in range(16):  # a bound, not a rule: these wrappers do not nest this deep
        if isinstance(target, functools.partial):
            target = target.func
        elif isinstance(target, staticmethod) or inspect.ismethod(target):
            target = target.__func__
        elif isinstance(target, _LRU_CACHE):
            target = target.__wrapped__
        else:
            break
    return target


def _is_async(func: Callable[..., Any]) -> bool:
    """Whether ``func`` is declared async, so calling it would run nothing.

    This reads the declaration (``async def``, or a function marked with
    ``inspect.markcoroutinefunction``) through a partial, a staticmethod, a bound method,
    ``functools.lru_cache`` and a callable object (``_declared``). A wrapper that hides
    the declaration is not seen here; what it returns is (``_answer``).
    """
    target = _declared(func)
    if inspect.iscoroutinefunction(target) or inspect.isasyncgenfunction(target):
        return True
    call = _declared(getattr(type(target), "__call__", None))
    return inspect.iscoroutinefunction(call) or inspect.isasyncgenfunction(call)


def _is_generator(func: Callable[..., Any]) -> bool:
    """Whether ``func`` is declared as a generator function, so calling it would run nothing.

    Read the way ``_is_async`` reads ``async def``. A wrapper that hides the declaration is
    not seen here; what it returns is (``_answer``).
    """
    target = _declared(func)
    call = _declared(getattr(type(target), "__call__", None))
    return inspect.isgeneratorfunction(target) or inspect.isgeneratorfunction(call)


def _cached(func: Callable[..., Any]) -> bool:
    """Whether ``func`` is a ``functools.lru_cache`` of a function, a method of one, or
    a ``staticmethod`` object around either: the forms of a cached tool the gate can
    judge. Every argument is then hashed."""
    target: Any = func
    for _ in range(16):  # a bound, not a rule: these wrappers do not nest this deep
        if isinstance(target, staticmethod) or inspect.ismethod(target):
            target = target.__func__
        else:
            break
    return isinstance(target, _LRU_CACHE)


def _answer(name: str, result: Any) -> str:
    """The text for what an allowed tool returned.

    The tool has been called by now: its own body ran, whatever it returned.

    A coroutine is not a result: the gate did not run the coroutine, and says so instead
    of answering with the object's ``repr``. One that was never started is closed, which
    runs none of its code and only spares the "never awaited" warning. One that has
    started belongs to whoever is running it and is left alone.

    An iterator is not a result either: a generator, an async generator, and any other
    object with ``__next__`` or ``__anext__`` (``iter(...)``, ``map(...)``, an open file).
    The gate does not consume one, since taking its items is running more code, so the
    answer says that the tool ran and that its result is not available. It never says the
    tool did not run: the function that returned the iterator did. The iterator is left
    alone. Any other value, a future or another awaitable included, is answered as its
    ``str``, as before.
    """
    if inspect.iscoroutine(result):
        if inspect.getcoroutinestate(result) == "CORO_CREATED":
            result.close()
        return (
            f"Tool `{name}` returned a coroutine, not a result: "
            "this gate calls tools synchronously and did not run it"
        )
    if isinstance(result, (Iterator, AsyncIterator)):
        kind = (
            "a generator"
            if inspect.isgenerator(result)
            else "an async generator"
            if inspect.isasyncgen(result)
            else "an iterator"
        )
        return (
            f"Tool `{name}` ran and returned {kind}, which this gate does not consume: "
            "its result is not available"
        )
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

    The entry is bound to ``proposal``: an object as it is, anything else as
    ``{"input": proposal}``. Arguments on an item the SDK did not validate can be
    something no fingerprint can be taken of (bytes, a number that is not finite, an
    object with a key that is not a string). The refusal is the same one either way, so
    the entry is then bound to ``_NOT_JSON_DATA`` and carries the same reason.
    """
    if not record:
        return plain
    if not _json_data(proposal):
        proposal = _NOT_JSON_DATA
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
    before the policy runs, read through a partial, a staticmethod, a bound method and
    ``functools.lru_cache``. An ordinary function that returns a coroutine is answered
    with the coroutine not run, and one that returns a generator or another iterator as
    having run with no result available (``_answer``). A tool cached with
    ``functools.lru_cache`` is refused before the policy runs when an argument is a list
    or an object, which the cache cannot hash.

    A refusal names a tool in ``tools`` by its name. Any other name is used as sent only
    if it is 1 to 64 printable characters with no backtick; otherwise the refusal, and
    its audit entry, describe it by its length (``_shown``). A name that is not a string
    is an unknown tool.

    A tool that raises propagates by default. With ``answer_tool_errors=True`` it is answered with
    ``Tool `name` raised <ExceptionClass>``, where the class is the exception's nearest
    built-in one: never its message, and never a name an application chose. The traceback
    goes to the ``recusal_openai`` logger; a logging handler that fails does not change
    the answer. Only ``Exception`` is caught, so a keyboard interrupt still propagates.

    ``record`` is the recording arguments for ``decide``: ``audit=`` with ``surface=``,
    and optionally ``tool_use_id=``, ``actor=``, ``control=`` and ``allow_on_pass=``
    (which records a passing call as ``allow`` instead of ``defer`` and changes nothing
    about what runs). Without ``audit=``, or with ``audit=None``, nothing is recorded, so
    a loop that puts the decision on a log (``audited_function_calls``) decides,
    and refuses, exactly as this one does. The entry for a judged call carries the
    fingerprint of the bound invocation the tool is called with, defaults included; the
    entry for a call refused before judgment is bound to the arguments as proposed: the
    decoded object, or the text as sent when it is the text that was refused.

    A keyword in ``record`` that is not one of those raises ``TypeError`` (``fail_closed``
    among them: this gate always fails closed), as does ``audit=`` without ``surface=``,
    and ``surface=``, ``tool_use_id=``, ``actor=`` or ``control=`` with a value when
    ``audit=`` is not passed at all. With ``audit=None`` passed explicitly those keywords
    are accepted and not used. A ``surface=``, ``control=`` or ``tool_use_id=`` that
    ``decide`` would refuse (an empty one, a reserved surface, a control that is not a
    dict), and a ``tool_use_id=`` of more than 256 characters, raise ``ValueError``
    naming the keyword. All of these raise before the call is looked at.

    Whether the tool's signature can be judged is asked of the kernel
    (``recusal.invocation.unsupported_signature``): a reason it gives, and any exception
    it raises, refuses the call before the policy runs.
    """
    return _judge(
        name, arguments, tools, policy, answer_tool_errors, _recording(record, _CALL_KEYWORDS)
    )


def _judge(
    name: str,
    arguments: Any,
    tools: Tools,
    policy: Policy,
    answer_tool_errors: bool,
    record: dict,
) -> str:
    """``judge_and_run`` with recording arguments ``_recording`` has already checked: the
    loops check the turn's once and then judge each call here."""
    shown = _shown(name, tools)
    if not isinstance(arguments, str):
        detail = "arguments are not a string"
        return _unjudgeable(
            shown, f"Recusal refused `{shown}`: {detail}", detail, arguments, record
        )
    try:
        args = json.loads(arguments or "{}", parse_constant=_not_json, parse_float=_finite)
    except _NotFinite:
        detail = "a number in the arguments is not finite"
        return _unjudgeable(
            shown, f"Recusal refused `{shown}`: {detail}", detail, arguments, record
        )
    except RecursionError:
        detail = _TOO_DEEP
        return _unjudgeable(
            shown, f"Recusal refused `{shown}`: {detail}", detail, arguments, record
        )
    except ValueError:
        detail = "arguments are not valid JSON"
        return _unjudgeable(
            shown, f"Recusal refused `{shown}`: {detail}", detail, arguments, record
        )
    # nothing nested deeper is fingerprinted, copied or handed to a policy, each of which
    # walks the arguments and would fail at a depth that depends on the caller's stack
    if _too_deep(args):
        detail = _TOO_DEEP
        return _unjudgeable(
            shown, f"Recusal refused `{shown}`: {detail}", detail, arguments, record
        )
    if not isinstance(args, dict):
        detail = "arguments are not a JSON object"
        return _unjudgeable(shown, f"Recusal refused `{shown}`: {detail}", detail, args, record)
    if not isinstance(name, str) or name not in tools:
        return _unjudgeable(
            shown, f"Recusal refused: unknown tool {shown!r}", "unknown tool", args, record
        )
    func = tools[name]
    if _is_async(func):
        detail = "it is declared async, and this gate calls tools synchronously"
        return _unjudgeable(shown, f"Recusal refused `{shown}`: {detail}", detail, args, record)
    if _is_generator(func):
        detail = "it is a generator function, and calling it would run none of its body"
        return _unjudgeable(shown, f"Recusal refused `{shown}`: {detail}", detail, args, record)
    try:
        why = _unsupported_signature(func)
    except Exception:  # noqa: BLE001, a signature the kernel cannot read is not judged
        why = "its signature cannot be inspected"
    if why is not None:
        detail = f"its signature cannot be judged ({why})"
        return _unjudgeable(shown, f"Recusal refused `{shown}`: {detail}", detail, args, record)
    signature = inspect.signature(func)
    try:
        bound = signature.bind(**args)
    except TypeError:  # its text can quote a key the model chose, so it is not used
        detail = _unbound(signature.parameters, args)
        return _unjudgeable(shown, f"Recusal refused `{shown}`: {detail}", detail, args, record)
    bound.apply_defaults()
    invocation = dict(bound.arguments)
    if _cached(func) and any(type(value) in (list, dict) for value in invocation.values()):
        detail = (
            "it is cached with `functools.lru_cache`, which cannot take a list or an "
            "object as an argument"
        )
        return _unjudgeable(shown, f"Recusal refused `{shown}`: {detail}", detail, args, record)
    try:
        judged = _detached(invocation)
    except (TypeError, RecursionError):  # the policy has not run: a refusal, on the record
        detail = "the invocation cannot be copied as JSON data for the policy"
        return _unjudgeable(shown, f"Recusal refused `{shown}`: {detail}", detail, args, record)
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
        shown = _shown(name, tools)
        return _unjudgeable(
            shown, f"Recusal refused `{shown}`: {refused}", refused, arguments, record
        )
    return _judge(name, arguments, tools, policy, answer_tool_errors, record)


def _require_id(call_id: Any, what: str) -> str:
    """The SDK does not require an id on the answer; the adapter does, or the answer is lost.

    The id is the model's text, and it goes back in the answer and onto the audit entry
    exactly as sent. One longer than ``_ID_MAX`` characters is refused here, before
    anything in the turn runs, so neither can be grown without bound.
    """
    if not isinstance(call_id, str) or not call_id:
        raise ValueError(f"Recusal refused {what}: it has no call id to answer")
    if len(call_id) > _ID_MAX:
        raise ValueError(
            f"Recusal refused {what}: its call id is {len(call_id)} characters long; "
            f"the gate takes a call id of at most {_ID_MAX}"
        )
    return call_id


def _id_shown(call_id: Any) -> str:
    """A call id as a message the gate raises carries it: quoted if it is a string of 1 to
    ``_ID_MAX`` printable characters with no backtick, and otherwise described by its
    length, the rule ``_shown`` applies to a tool name."""
    if not isinstance(call_id, str):
        return "<a call id that is not a string>"
    if 0 < len(call_id) <= _ID_MAX and call_id.isprintable() and "`" not in call_id:
        return repr(call_id)
    return f"<a call id of {len(call_id)} characters, not shown>"


def _ids_shown(call_ids: list) -> str:
    """Up to four call ids for a message, each as ``_id_shown`` gives it, and how many more."""
    shown = ", ".join(_id_shown(call_id) for call_id in call_ids[:4])
    more = len(call_ids) - 4
    return f"[{shown}{f', and {more} more' if more > 0 else ''}]"


def _for_call(record: dict, call_id: str) -> dict:
    """The recording arguments for one call of a turn: the turn's, plus the call's own id."""
    return {**record, "tool_use_id": call_id} if record else {}


def _check_ids(call_ids: list[str]) -> None:
    """Refuse a turn with a duplicate id BEFORE any call in it runs: the turn's answers are
    keyed by id, so a duplicate could not be answered once, and a refusal raised after
    dispatch would leave a side effect behind for the caller's retry to repeat."""
    if len(set(call_ids)) != len(call_ids):
        raise ValueError(f"Recusal refused the turn: duplicate call ids {_repeated(call_ids)}")


def _repeated(call_ids: list[str]) -> str:
    """The call ids that occur more than once, in the order they first repeat, for a message."""
    seen: set[str] = set()
    repeated: list[str] = []
    for call_id in call_ids:
        if call_id in seen and call_id not in repeated:
            repeated.append(call_id)
        seen.add(call_id)
    return _ids_shown(repeated)


def _check_answered_once(call_ids: list[str], answered: list[str]) -> None:
    """Every proposed call gets exactly one answer, and no answer invents a call."""
    if sorted(call_ids) != sorted(answered):
        raise ValueError(
            f"Recusal refused the turn: calls {_ids_shown(sorted(call_ids))} but answers "
            f"{_ids_shown(sorted(answered))}"
        )
    if len(set(call_ids)) != len(call_ids):
        raise ValueError(f"Recusal refused the turn: duplicate call ids {_repeated(call_ids)}")


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
    does not take, and ValueError for a call without an id, a call id of more than 256
    characters, a duplicate id, or a ``surface=`` or ``control=`` that ``recusal.decide``
    would refuse. Pass the returned list as the
    next request's ``input``. A ``function_call`` item that carries a ``namespace``, one
    whose ``status`` is not ``completed`` (no status passes), and every function call of
    a response whose ``status`` is not ``completed`` (no status passes) are refused with a
    reason under their own ids, before the policy runs.
    ``allow_hosted_records=True`` lets a finished item of a type in ``HOSTED_RECORD_TYPES``
    pass: the provider produced it on its own side, this loop can neither judge nor stop
    it, so it is not answered and not put on the record. ``answer_tool_errors=True``
    answers a tool that raises instead of letting the exception leave the gate; see
    ``judge_and_run``.
    ``record`` is the recording arguments for the turn: ``audit=`` with ``surface=``, and
    optionally ``actor=``, ``control=`` and ``allow_on_pass=``; see ``judge_and_run``.
    Each call is recorded under its own ``call_id``, so ``tool_use_id=`` is not taken.
    ``audit=None`` passed explicitly is the plain gate, which records nothing and with
    which ``surface=``, ``actor=`` and ``control=`` are accepted and not used, so a loop
    whose log is optional is one call. When ``audit=`` is not passed at all, ``surface=``,
    ``actor=`` or ``control=`` with a value raises TypeError, whether or not the turn has
    calls.
    A turn refused as a whole raises before anything is judged, so it runs nothing and
    writes no entry.
    """
    record = _recording(record)
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
        call_id = _require_id(
            getattr(call, "call_id", None), f"function_call {_shown(call.name, tools)!r}"
        )
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
    carries the legacy ``function_call`` field, a keyword it does not take or a non-int
    ``choice``, and ValueError for a call without an id, a call id of more than 256
    characters, a duplicate id, no choices, an out-of-range ``choice``, or a ``surface=``
    or ``control=`` that ``recusal.decide`` would refuse. Append the returned messages
    after the chosen assistant message in the next request. When the chosen choice's
    ``finish_reason`` is not ``stop``,
    ``tool_calls`` or ``function_call`` (``length``: the message was cut off), every tool
    call in it is refused with a reason under its own id, before the policy runs.
    ``record`` is the recording arguments for the turn, and ``answer_tool_errors`` the
    same keyword, as in ``gate_function_calls``.
    """
    record = _recording(record)
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
        call_id = _require_id(call.id, f"tool call {_shown(call.function.name, tools)!r}")
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
