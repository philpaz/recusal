"""
Offline OpenAI API example: every gated function call in your own loop on an audit log.

examples/openai_responses_gate.py refuses unsafe function calls in a loop the application
owns. This example adds the record: each call the model proposes, run or refused, is one
hash-chained entry on a Recusal AuditLog, labeled with the surface
``openai.responses.function_call`` or ``openai.chat_completions.tool_call`` and the API's
own call id, so a quiet edit, a reordering, or a dropped entry is detectable with
``recusal audit verify``. The log holds a SHA-256 fingerprint of each call's arguments,
never the arguments themselves.

The audited loop is the ``recusal-openai`` package (``adapters/openai`` in this repository):
``audited_function_calls`` and ``audited_tool_calls`` are imported from it. They call
``gate_function_calls`` and ``gate_tool_calls`` with the recording arguments passed through
to ``recusal.decide``, which writes the entry before returning. If the log cannot be
written the decision becomes a deny, so nothing runs unrecorded.

What each entry is bound to:

- a call that is judged (run, or refused by the policy): the invocation the tool is called
  with, bound to its signature with defaults applied, so a default the model omitted is on
  the record at the value that runs;
- a call refused before judgment (arguments that are not a JSON object, an unknown tool, a
  signature that cannot be judged, arguments that do not bind): the arguments as the model
  proposed them.

What is not on the log: a turn refused as a whole. An output item of a type the adapter
does not handle, a call without an id and a duplicate id all raise to the application
before any call is judged, so nothing in that turn runs and no entry is written. Hosted
tools run on the provider's side and never reach this loop.

Requires Python 3.10+ (the SDK's minimum) and is tested with exactly openai==3.34.0.
Recusal itself remains Python 3.9+ and has no OpenAI dependency. No API key or network is
used, and run_sql only records SQL text.

Run after installing:
    pip install openai==3.34.0
    python examples/openai_responses_audit.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Optional

OPENAI_VERSION = "3.34.0"

if sys.version_info < (3, 10):
    print(
        "openai_responses_audit.py requires Python 3.10+ "
        f"(openai {OPENAI_VERSION}); "
        f"this is Python {sys.version_info[0]}.{sys.version_info[1]}.",
        file=sys.stderr,
    )
    raise SystemExit(2)

try:
    import openai  # noqa: F401, the package this example demonstrates needs it
except ImportError as exc:
    raise SystemExit(f"Install the example dependency first: openai=={OPENAI_VERSION}") from exc

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
# In a checkout the adapter is read from the tree; installed, `pip install
# recusal-openai` provides the same package and this path does not exist.
sys.path.insert(0, os.path.join(_ROOT, "adapters", "openai", "src"))

from recusal_openai import (  # noqa: E402
    SURFACE_CHAT,
    SURFACE_RESPONSES,
    Policy,
    Tools,
    audited_function_calls,
    audited_tool_calls,
)

from examples.openai_responses_gate import (  # noqa: E402
    CALLS,
    EXECUTED,
    TOOLS,
    completion_with,
    function_call,
    response_with,
    sql_scope_policy,
    tool_call,
)
from recusal.audit import AuditLog  # noqa: E402

__all__ = [
    "SURFACE_CHAT",
    "SURFACE_RESPONSES",
    "audited_function_calls",
    "audited_tool_calls",
]


def run_calls(
    calls: list[tuple[str, Any, str]],
    log: AuditLog,
    *,
    policy: Policy = sql_scope_policy,
    tools: Optional[Tools] = None,
    chat: bool = False,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, str], list[tuple[str, dict[str, Any]]]]:
    """Propose the calls through one API shape behind the audited gate; return each call's
    answer by id and what ran."""
    EXECUTED.clear()
    registry = TOOLS if tools is None else tools
    if chat:
        completion = completion_with([tool_call(*call) for call in calls])
        outputs = audited_tool_calls(
            completion, registry, policy, log, actor=actor, control=control
        )
        answers = {out["tool_call_id"]: out["content"] for out in outputs}
    else:
        response = response_with([function_call(*call) for call in calls])
        outputs = audited_function_calls(
            response, registry, policy, log, actor=actor, control=control
        )
        answers = {out["call_id"]: out["output"] for out in outputs}
    return answers, list(EXECUTED)


def verify_with_cli(path: str) -> tuple[int, str]:
    """Run ``recusal audit verify`` on the log the way an auditor would; (exit code, text)."""
    result = subprocess.run(
        [sys.executable, "-m", "recusal", "audit", "verify", path],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, (result.stdout + result.stderr).strip()


def tamper(path: str, tool_use_id: str, decision: str = "allow") -> None:
    """Rewrite one entry's recorded decision in place, the quiet edit the chain must catch."""
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    for index, line in enumerate(lines):
        entry = json.loads(line)
        if entry["action"].get("tool_use_id") == tool_use_id:
            entry["action"]["decision"] = decision
            lines[index] = json.dumps(entry, sort_keys=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "openai-audit.jsonl")
        log = AuditLog(path=path)
        answers, executed = run_calls(CALLS, log, actor="demo-agent")

        print("RECUSAL - OpenAI function calls in your own loop, every call on the record\n")
        for _, _, call_id in CALLS:
            ran = "refused" if answers[call_id].startswith("Recusal") else "ran"
            print(f"  {call_id} {ran}")
        print(f"  executed: {len(executed)} call(s)\n")

        by_id = {entry["action"]["tool_use_id"]: entry for entry in log.entries}
        for name, _, call_id in CALLS:
            entry = by_id[call_id]
            action = entry["action"]
            print(
                f"  {call_id} {action['surface']} {action['tool']:<8} "
                f"{action['decision']:<5} {entry['actor']} {entry['hash'][:12]}..."
            )

        code, text = verify_with_cli(path)
        print(f"\n  verify intact:   exit {code}  {text}")

        tamper(path, "call_2")  # the refused DELETE now reads as allowed
        code, text = verify_with_cli(path)
        print(f"  verify tampered: exit {code}  {text}")


if __name__ == "__main__":
    main()
