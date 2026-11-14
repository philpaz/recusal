"""Prove the INSTALLED recusal-openai wheel gates real OpenAI SDK response objects.

Run it from a directory outside the repository, with an interpreter that has the built
wheels installed:

    python /path/to/adapters/openai/smoke_test.py

It fails if the package it imports is the source tree and not an installed distribution:
a check run from the checkout would prove nothing about the wheel. The responses are built
with the SDK's own ``model_validate``; no API key or network is used.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from importlib import metadata
from pathlib import Path

# Python puts this script's directory first on the import path. Drop it, so nothing in
# the source tree can stand in for the installed package.
_HERE = Path(__file__).resolve().parent
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != _HERE]

import recusal_openai  # noqa: E402
from openai.types.chat import ChatCompletion  # noqa: E402
from openai.types.responses import Response  # noqa: E402
from recusal_openai import (  # noqa: E402
    SURFACE_CHAT,
    SURFACE_RESPONSES,
    audited_function_calls,
    audited_tool_calls,
    gate_function_calls,
    gate_tool_calls,
)

import recusal  # noqa: E402
from recusal import Finding  # noqa: E402
from recusal.audit import AuditLog, verify  # noqa: E402

EXECUTED: list[str] = []
CALLS = [("c1", "SELECT 1"), ("c2", "DELETE FROM t")]


def run_sql(sql: str) -> str:
    """Record SQL text without connecting to a database."""
    EXECUTED.append(sql)
    return f"recorded: {sql}"


TOOLS = {"run_sql": run_sql}


def policy(tool_name, tool_input):
    if str(tool_input.get("sql", "")).upper().startswith("DELETE"):
        return [Finding.fail("no_deletes", severity="CRITICAL", message="DELETE is not allowed")]
    return []


def response() -> Response:
    return Response.model_validate(
        {
            "id": "resp_offline",
            "object": "response",
            "created_at": 0,
            "model": "offline",
            "output": [
                {
                    "type": "function_call",
                    "id": f"fc_{ident}",
                    "call_id": ident,
                    "name": "run_sql",
                    "arguments": json.dumps({"sql": sql}),
                    "status": "completed",
                }
                for ident, sql in CALLS
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
    )


def completion() -> ChatCompletion:
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
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": ident,
                                "type": "function",
                                "function": {
                                    "name": "run_sql",
                                    "arguments": json.dumps({"sql": sql}),
                                },
                            }
                            for ident, sql in CALLS
                        ],
                    },
                }
            ],
        }
    )


def main() -> None:
    for module in (recusal, recusal_openai):
        origin = Path(module.__file__).resolve()
        assert "site-packages" in origin.parts, f"{module.__name__} is not installed: {origin}"
    assert metadata.version("recusal-openai") == recusal_openai.__version__
    requires = metadata.requires("recusal-openai") or []
    assert any(r.startswith("recusal") and "openai" not in r for r in requires), requires

    # Negative control: a policy that refuses nothing runs the DELETE, so the refusals
    # below are the policy's, reached through the gate.
    EXECUTED.clear()
    gate_function_calls(response(), TOOLS, lambda *_: [])
    assert EXECUTED == ["SELECT 1", "DELETE FROM t"], EXECUTED

    EXECUTED.clear()
    answers = {a["call_id"]: a["output"] for a in gate_function_calls(response(), TOOLS, policy)}
    assert EXECUTED == ["SELECT 1"], EXECUTED
    assert answers["c1"] == "recorded: SELECT 1" and "DELETE is not allowed" in answers["c2"]

    EXECUTED.clear()
    messages = {
        m["tool_call_id"]: m["content"] for m in gate_tool_calls(completion(), TOOLS, policy)
    }
    assert EXECUTED == ["SELECT 1"], EXECUTED
    assert "DELETE is not allowed" in messages["c2"], messages

    for audited, proposal, surface in (
        (audited_function_calls, response(), SURFACE_RESPONSES),
        (audited_tool_calls, completion(), SURFACE_CHAT),
    ):
        EXECUTED.clear()
        with tempfile.TemporaryDirectory() as folder:
            log = AuditLog(path=os.path.join(folder, "audit.jsonl"))
            audited(proposal, TOOLS, policy, log)
            assert EXECUTED == ["SELECT 1"], EXECUTED
            by_id = {entry["action"]["tool_use_id"]: entry["action"] for entry in log.entries}
            assert by_id["c2"]["decision"] == "deny" and by_id["c2"]["surface"] == surface, by_id
            assert by_id["c1"]["decision"] != "deny", by_id
            intact, problems = verify(log.entries)
            assert intact, problems

    print(
        f"recusal-openai {recusal_openai.__version__} with recusal "
        f"{recusal.__version__}: installed wheel gates and records"
    )


if __name__ == "__main__":
    main()
