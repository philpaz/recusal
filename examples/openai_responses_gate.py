"""
Offline OpenAI API example: gate each function call before your own loop runs it.

Not every agent uses a framework. An application that calls the OpenAI API directly reads
the function calls out of the response, runs them, and sends the results back. That loop
belongs to the application, which makes it the cleanest place to put the gate. This example
covers both shapes of the loop with one decision function:

- the Responses API: ``function_call`` output items, answered with ``function_call_output``
  input items (``gate_function_calls``);
- Chat Completions: ``tool_calls`` on the assistant message, answered with ``role: "tool"``
  messages (``gate_tool_calls``).

For every call the model proposes, the adapter decodes the arguments the way the tool would
receive them, judges the call with ``recusal.claude_code.decide``, runs the tool only if it
is not denied, and returns exactly one answer carrying that call's own id. Arguments that are
not valid JSON or not a JSON object, a tool the application does not have, and a policy that
raises are all refused with a reason, never run.

Default-deny on what the loop does not understand: an output item or tool call of a type
this adapter does not handle (``custom_tool_call``, a hosted tool call, a type added in a
later SDK release) raises, naming the type, instead of being skipped. And because the SDK's
answer types do not require ``call_id`` (checked on openai==3.34.0: every key of
``FunctionCallOutput`` and ``ChatCompletionToolMessageParam`` is optional), the adapter
refuses a call without an id and checks that every call got exactly one answer.

Hosted tools (web search, file search, code interpreter, image generation, remote MCP) run
on the provider's side and never reach this loop, so nothing here can gate them.

Requires Python 3.10+ (the SDK's minimum) and is tested with exactly openai==3.34.0. The
Response and ChatCompletion objects are built with the SDK's own ``model_validate``, so they
are real SDK types; no API key or network is used, and run_sql only records SQL text.
Recusal itself remains Python 3.9+ and has no OpenAI dependency.

Run after installing:
    pip install openai==3.34.0
    python examples/openai_responses_gate.py
"""

from __future__ import annotations

import copy
import inspect
import json
import os
import sys
import typing
from typing import Any, Callable, List, Optional

OPENAI_VERSION = "3.34.0"

if sys.version_info < (3, 10):
    print(
        "openai_responses_gate.py requires Python 3.10+ "
        f"(openai {OPENAI_VERSION}); "
        f"this is Python {sys.version_info[0]}.{sys.version_info[1]}.",
        file=sys.stderr,
    )
    raise SystemExit(2)

try:
    from openai.types.chat import ChatCompletion
    from openai.types.responses import Response
except ImportError as exc:
    raise SystemExit(f"Install the example dependency first: openai=={OPENAI_VERSION}") from exc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.sql_scope_policy import policy as sql_scope_policy  # noqa: E402
from recusal.claude_code import decide  # noqa: E402

Policy = Callable[[str, dict], List[Any]]
Tools = dict[str, Callable[..., Any]]
EXECUTED: list[tuple[str, dict[str, Any]]] = []

#: Response output items that cannot run anything, so the gate lets them pass. Everything
#: else (custom_tool_call, hosted tool calls, a type a later SDK adds) raises.
PASSIVE_OUTPUT_TYPES = frozenset({"message", "reasoning"})


def echo(text: str) -> str:
    """Return harmless text."""
    EXECUTED.append(("echo", {"text": text}))
    return text


def run_sql(sql: str) -> str:
    """Record SQL text without connecting to a database."""
    EXECUTED.append(("run_sql", {"sql": sql}))
    return f"recorded SQL: {sql}"


def charge(amount: int = 100) -> str:
    """Record a hypothetical charge with a default the model may omit; contacts nothing."""
    EXECUTED.append(("charge", {"amount": amount}))
    return "recorded"


TOOLS: Tools = {"echo": echo, "run_sql": run_sql, "charge": charge}

# --- the supported invocation: an explicit type set, defaults completed before judgment -------

_SCALARS = (str, int, float, bool)
_UNION_TYPE = type(int | None)  # the PEP 604 union, without shadowing any `types` module


def _supported_annotation(annotation: Any) -> bool:
    """True for an annotation whose preparation is a pure function of the value: ``str``,
    ``int``, ``float``, ``bool``, ``Optional`` of one, ``list[T]`` and ``dict[str, T]`` with
    ``T`` supported in turn. Everything else (no annotation, ``Any``, a model, an ``Enum``, an
    ``Annotated``, a dataclass, a bare ``list``/``dict``) is unsupported."""
    if annotation in _SCALARS:
        return True
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is typing.Union or origin is _UNION_TYPE:
        rest = [a for a in args if a is not type(None)]
        return len(rest) == 1 and _supported_annotation(rest[0])
    if origin is list:
        return len(args) == 1 and _supported_annotation(args[0])
    if origin is dict:
        return len(args) == 2 and args[0] is str and _supported_annotation(args[1])
    return False


def _json_data(value: Any) -> bool:
    """True for a value in the JSON domain (what a default may be)."""
    if value is None or isinstance(value, _SCALARS):
        return True
    if isinstance(value, list):
        return all(_json_data(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_data(v) for k, v in value.items())
    return False


def _unsupported_signature(func: Callable[..., Any], skip: tuple[str, ...] = ()) -> Optional[str]:
    """Why this callable's parameters cannot all be judged, or None if they can."""
    try:
        parameters = inspect.signature(func).parameters
        # resolves string annotations (PEP 563) and keeps Annotated so it can be refused
        hints = typing.get_type_hints(func, include_extras=True)
    except Exception:  # noqa: BLE001, an uninspectable callable cannot be judged
        return "its signature cannot be inspected"
    for name, parameter in parameters.items():
        if name in skip:
            continue
        annotation = hints.get(name, parameter.empty)
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            return f"`{name}` is variadic; every argument the tool receives must be judged by name"
        if parameter.kind is parameter.POSITIONAL_ONLY:
            return f"`{name}` is positional-only and cannot be judged by name"
        if not _supported_annotation(annotation):
            return f"`{name}` is not annotated with a supported data type"
        if parameter.default is not parameter.empty and not _json_data(parameter.default):
            return f"`{name}` has a default outside the JSON domain"
    return None


def judge_and_run(name: str, arguments: str | None, tools: Tools, policy: Policy) -> str:
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
    """
    try:
        args = json.loads(arguments or "{}")
    except ValueError:
        return f"Recusal refused `{name}`: arguments are not valid JSON"
    if not isinstance(args, dict):
        return f"Recusal refused `{name}`: arguments are not a JSON object"
    if name not in tools:
        return f"Recusal refused: unknown tool {name!r}"
    func = tools[name]
    why = _unsupported_signature(func)
    if why is not None:
        return f"Recusal refused `{name}`: its signature cannot be judged ({why})"
    try:
        bound = inspect.signature(func).bind(**args)
    except TypeError as exc:  # the message names parameters only, never values
        return f"Recusal refused `{name}`: arguments do not bind to the tool ({exc})"
    bound.apply_defaults()
    invocation = dict(bound.arguments)
    try:
        decision, reason = decide(name, copy.deepcopy(invocation), policy, safe_errors=True)
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


def gate_function_calls(response: Response, tools: Tools, policy: Policy) -> list[dict[str, Any]]:
    """Responses API: judge every ``function_call`` item; return ``function_call_output`` items.

    Raises TypeError for an output item type this adapter does not handle, and ValueError for
    a call without an id. Pass the returned list as the next request's ``input``.
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
            "output": judge_and_run(name, arguments, tools, policy),
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
    completion: ChatCompletion, tools: Tools, policy: Policy, *, choice: Optional[int] = None
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
            "content": judge_and_run(name, arguments, tools, policy),
        }
        for call_id, name, arguments in calls
    ]
    _check_answered_once(call_ids, [o["tool_call_id"] for o in outputs])
    return outputs


def dispatch_ungated(response: Response, tools: Tools) -> list[dict[str, Any]]:
    """The naive loop this example replaces: run every function call as proposed."""
    outputs = []
    for item in response.output:
        if getattr(item, "type", None) == "function_call":
            args = json.loads(item.arguments or "{}")
            # the naive loop: Python fills omitted defaults after the call is dispatched
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": str(tools[item.name](**args)),
                }
            )
    return outputs


# --- offline fixtures: real SDK objects, no network ------------------------------------------


def response_with(output: list[dict[str, Any]]) -> Response:
    """A Response the SDK itself validated, with the given output items; no model is called."""
    return Response.model_validate(
        {
            "id": "resp_offline",
            "object": "response",
            "created_at": 0,
            "model": "offline",
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
    )


def function_call(name: str, arguments: Any, call_id: str) -> dict[str, Any]:
    """A ``function_call`` output item; ``arguments`` is JSON-encoded unless already a string."""
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
        "status": "completed",
    }


def completion_with(
    tool_calls: list[dict[str, Any]], *alternatives: list[dict[str, Any]]
) -> ChatCompletion:
    """A ChatCompletion the SDK itself validated, proposing the given tool calls; each
    further list is one more alternative choice (what ``n > 1`` returns)."""
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl_offline",
            "object": "chat.completion",
            "created": 0,
            "model": "offline",
            "choices": [
                {
                    "index": index,
                    "finish_reason": "tool_calls",
                    "message": {"role": "assistant", "content": None, "tool_calls": calls},
                }
                for index, calls in enumerate([tool_calls, *alternatives])
            ],
        }
    )


def tool_call(name: str, arguments: Any, call_id: str) -> dict[str, Any]:
    """A Chat Completions function tool call; ``arguments`` is JSON-encoded unless a string."""
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
        },
    }


CALLS: list[tuple[str, Any, str]] = [
    ("run_sql", {"sql": "SELECT * FROM audit_log"}, "call_1"),
    ("run_sql", {"sql": "DELETE FROM audit_log"}, "call_2"),
    ("run_sql", "{not json", "call_3"),
    ("drop_db", {"name": "prod"}, "call_4"),
    ("run_sql", ["SELECT 1"], "call_5"),
]


def main() -> None:
    print("RECUSAL - OpenAI function calls gated in your own loop (offline)\n")
    response = response_with([function_call(*call) for call in CALLS])

    EXECUTED.clear()
    print("Responses API:")
    for out in gate_function_calls(response, TOOLS, sql_scope_policy):
        print(f"  {out['call_id']} -> {out['output']}")
    print(f"  executed: {[args.get('sql') for _, args in EXECUTED]}\n")

    EXECUTED.clear()
    print("Chat Completions:")
    completion = completion_with([tool_call(*call) for call in CALLS])
    for out in gate_tool_calls(completion, TOOLS, sql_scope_policy):
        print(f"  {out['tool_call_id']} -> {out['content']}")
    print(f"  executed: {[args.get('sql') for _, args in EXECUTED]}\n")

    EXECUTED.clear()
    runnable = response_with([function_call(*call) for call in CALLS[:2]])
    dispatch_ungated(runnable, TOOLS)
    print(f"UNGATED executed: {[args.get('sql') for _, args in EXECUTED]}")

    custom = response_with(
        [
            {
                "type": "custom_tool_call",
                "id": "ct_1",
                "call_id": "call_9",
                "name": "x",
                "input": "raw",
            }
        ]
    )
    try:
        gate_function_calls(custom, TOOLS, sql_scope_policy)
    except TypeError as exc:
        print(f"custom_tool_call: {exc}")


if __name__ == "__main__":
    main()
