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

import json
import os
import sys
from typing import Any, Callable, List

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


TOOLS: Tools = {"echo": echo, "run_sql": run_sql}


def judge_and_run(name: str, arguments: str | None, tools: Tools, policy: Policy) -> str:
    """Decide one proposed call and run it if allowed; return the text the model gets back.

    The ONE decision path for both API shapes. The arguments are decoded exactly as the tool
    would receive them, so the policy judges what would run, not a look-alike.
    """
    try:
        args = json.loads(arguments or "{}")
    except ValueError:
        return f"Recusal refused `{name}`: arguments are not valid JSON"
    if not isinstance(args, dict):
        return f"Recusal refused `{name}`: arguments are not a JSON object"
    if name not in tools:
        return f"Recusal refused: unknown tool {name!r}"
    try:
        decision, reason = decide(name, args, policy)
    except Exception as exc:  # noqa: BLE001, a buggy policy must not silently disable the gate
        return f"Recusal failed closed (policy error): {exc}"
    if decision == "deny":
        return reason
    return str(tools[name](**args))


def _require_id(call_id: Any, what: str) -> str:
    """The SDK does not require an id on the answer; the adapter does, or the answer is lost."""
    if not isinstance(call_id, str) or not call_id:
        raise ValueError(f"Recusal refused {what}: it has no call id to answer")
    return call_id


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
    call_ids: list[str] = []
    outputs: list[dict[str, Any]] = []
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
        call_ids.append(call_id)
        outputs.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": judge_and_run(item.name, item.arguments, tools, policy),
            }
        )
    _check_answered_once(call_ids, [o["call_id"] for o in outputs])
    return outputs


def gate_tool_calls(
    completion: ChatCompletion, tools: Tools, policy: Policy
) -> list[dict[str, Any]]:
    """Chat Completions: judge every ``tool_calls`` entry; return ``role: "tool"`` messages.

    Raises TypeError for a tool call type this adapter does not handle (``custom``), and
    ValueError for a call without an id. Append the returned messages after the assistant
    message in the next request.
    """
    call_ids: list[str] = []
    outputs: list[dict[str, Any]] = []
    for choice in completion.choices:
        for call in choice.message.tool_calls or []:
            if call.type != "function":
                raise TypeError(
                    f"Recusal refused tool call of type {call.type!r}: "
                    "this adapter handles function tool calls only"
                )
            call_id = _require_id(call.id, f"tool call {call.function.name!r}")
            call_ids.append(call_id)
            outputs.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": judge_and_run(
                        call.function.name, call.function.arguments, tools, policy
                    ),
                }
            )
    _check_answered_once(call_ids, [o["tool_call_id"] for o in outputs])
    return outputs


def dispatch_ungated(response: Response, tools: Tools) -> list[dict[str, Any]]:
    """The naive loop this example replaces: run every function call as proposed."""
    outputs = []
    for item in response.output:
        if getattr(item, "type", None) == "function_call":
            args = json.loads(item.arguments or "{}")
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


def completion_with(tool_calls: list[dict[str, Any]]) -> ChatCompletion:
    """A ChatCompletion the SDK itself validated, proposing the given tool calls."""
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl_offline",
            "object": "chat.completion",
            "created": 0,
            "model": "offline",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
                }
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
