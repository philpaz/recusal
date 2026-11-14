"""
Offline OpenAI API example: gate each function call before your own loop runs it.

The gate is the ``recusal-openai`` package (``adapters/openai`` in this repository):
``gate_function_calls``, ``gate_tool_calls`` and ``judge_and_run`` are imported from it.
This file is the offline demonstration around it: SDK-validated responses built without
a network, three tools that only record, and the SQL scope policy.

Not every agent uses a framework. An application that calls the OpenAI API directly reads
the function calls out of the response, runs them, and sends the results back. That loop
belongs to the application, which makes it the cleanest place to put the gate. This example
covers both shapes of the loop with one decision function:

- the Responses API: ``function_call`` output items, answered with ``function_call_output``
  input items (``gate_function_calls``);
- Chat Completions: ``tool_calls`` on the assistant message, answered with ``role: "tool"``
  messages (``gate_tool_calls``).

For every call the model proposes, the adapter decodes the arguments the way the tool would
receive them, judges the call with ``recusal.decide``, runs the tool only if it
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
from typing import Any

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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
# In a checkout the adapter is read from the tree; installed, `pip install
# recusal-openai` provides the same package and this path does not exist.
sys.path.insert(0, os.path.join(_ROOT, "adapters", "openai", "src"))

from recusal_openai import (  # noqa: E402
    PASSIVE_OUTPUT_TYPES,
    Policy,
    Tools,
    gate_function_calls,
    gate_tool_calls,
    judge_and_run,
)

from examples.sql_scope_policy import policy as sql_scope_policy  # noqa: E402

__all__ = [
    "PASSIVE_OUTPUT_TYPES",
    "Policy",
    "Tools",
    "gate_function_calls",
    "gate_tool_calls",
    "judge_and_run",
]

EXECUTED: list[tuple[str, dict[str, Any]]] = []


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
