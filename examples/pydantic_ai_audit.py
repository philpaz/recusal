"""
Offline Pydantic AI example: every gated tool call on a tamper-evident audit log.

examples/pydantic_ai_gate.py refuses unsafe tool calls in a Pydantic AI agent. This example
adds the record: each call the gate sees, allowed or refused, is one hash-chained entry on
a Recusal AuditLog, labeled with the surface ``pydantic_ai.before_tool_execute`` and
Pydantic AI's own tool call id, so a quiet edit, a reordering, or a dropped entry is
detectable with ``recusal audit verify``. The log holds a SHA-256 fingerprint of each
call's arguments, never the arguments themselves.

The decision logic is the one in pydantic_ai_gate.py: ``RecusalAuditedGate(policy, log=log)``
and ``recusal_audited_before_tool_execute(policy, log)`` call the same ``_refusal`` with the
recording arguments passed through to ``recusal.decide``, which writes the entry before
returning. If the log cannot be written the decision becomes a deny, so nothing runs
unrecorded.

What each entry is bound to:

- a call that is judged (allowed or refused by the policy): the validated arguments the
  tool is called with;
- a call refused before judgment (arguments the gate cannot detach): the call as the model
  proposed it, since the validated arguments are not data a fingerprint can cover.

What is not on the log, because the hook never runs for it: a call whose arguments fail
Pydantic AI's validation (answered with a retry prompt, and the tool does not run), and
the tool calls of an agent another agent delegates to, unless that agent has its own
audited gate.

Requires Python 3.11+ (Pydantic AI's minimum) and is tested with exactly
pydantic-ai-slim==2.64.0. Recusal itself remains Python 3.9+ and has no Pydantic AI
dependency.

No model, API key or database is used. run_sql only records SQL text.

Run after installing:
    pip install pydantic-ai-slim==2.64.0
    python examples/pydantic_ai_audit.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Optional

PYDANTIC_AI_VERSION = "2.64.0"

if sys.version_info < (3, 11):
    print(
        "pydantic_ai_audit.py requires Python 3.11+ "
        f"(pydantic-ai-slim {PYDANTIC_AI_VERSION}); "
        f"this is Python {sys.version_info[0]}.{sys.version_info[1]}.",
        file=sys.stderr,
    )
    raise SystemExit(2)

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

try:
    from pydantic_ai.capabilities import Hooks
    from pydantic_ai.exceptions import ToolFailed
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import ToolDefinition
except ImportError as exc:
    raise SystemExit(
        f"Install the example dependency first: pydantic-ai-slim=={PYDANTIC_AI_VERSION}"
    ) from exc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.pydantic_ai_gate import (  # noqa: E402
    EXECUTED,
    Policy,
    RecusalGate,
    _refusal,
    responses_of,
    scripted_agent,
    sql_scope_policy,
)
from recusal.audit import AuditLog  # noqa: E402

#: The surface label every entry carries: the runtime hook that adjudicated.
SURFACE = "pydantic_ai.before_tool_execute"


def _recording(
    call: ToolCallPart, actor: Optional[str], control: Optional[dict[str, Any]], log: AuditLog
) -> dict[str, Any]:
    """The recording arguments for one call, built in one place for both forms of the gate.

    Pydantic AI generates a tool call id when the model sends none, so an entry normally
    carries one; a blank id is recorded as no id rather than refused."""
    ident = call.tool_call_id
    return {
        "audit": log,
        "surface": SURFACE,
        "tool_use_id": ident if isinstance(ident, str) and ident else None,
        "actor": actor,
        "control": control,
    }


def _proposal(call: ToolCallPart) -> Any:
    """The call's arguments as the model sent them, for the entry of a call refused before
    judgment. Never raises: an argument string that is not a JSON object is kept as text."""
    try:
        return call.args_as_dict()
    except Exception:  # noqa: BLE001, the refusal is recorded whatever the proposal looks like
        return call.args if isinstance(call.args, (str, dict)) else None


def recusal_audited_before_tool_execute(
    policy: Policy,
    log: AuditLog,
    *,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
):
    """Return a ``before_tool_execute`` hook that records, then gates, each call.

    For ``Hooks(before_tool_execute=...)``, which declares no ordering: list it last, as
    with the plain hook, or a later hook can change the arguments after they were recorded.
    """

    async def gate(ctx: Any, *, call: ToolCallPart, tool_def: ToolDefinition, args: dict) -> dict:
        reason = _refusal(
            call.tool_name,
            args,
            policy,
            proposal=_proposal(call),
            **_recording(call, actor, control, log),
        )
        if reason is not None:
            raise ToolFailed(reason)
        return args

    gate.recusal_gate = True  # type: ignore[attr-defined]
    return gate


@dataclass(kw_only=True)
class RecusalAuditedGate(RecusalGate):
    """The audited gate as a capability: ``RecusalAuditedGate(policy, log=log)``.

    A ``RecusalGate`` (so ``is_gated`` sees it, and it keeps the ``innermost`` position)
    whose every decision is one entry on ``log``.
    """

    log: AuditLog
    actor: Optional[str] = None
    control: Optional[dict[str, Any]] = None

    async def before_tool_execute(
        self, ctx: Any, *, call: ToolCallPart, tool_def: ToolDefinition, args: dict
    ) -> dict:
        reason = _refusal(
            call.tool_name,
            args,
            self.policy,
            proposal=_proposal(call),
            **_recording(call, self.actor, self.control, self.log),
        )
        if reason is not None:
            raise ToolFailed(reason)
        return args


def run_calls(
    calls: list[tuple[str, Any, str]],
    log: AuditLog,
    *,
    policy: Policy = sql_scope_policy,
    via_hooks: bool = False,
    tools: Optional[list] = None,
    actor: Optional[str] = None,
    control: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, tuple[str, Any]], list[tuple[str, dict[str, Any]]]]:
    """Propose the calls to an agent behind the audited gate; return answers and what ran."""
    EXECUTED.clear()
    if via_hooks:
        hook = recusal_audited_before_tool_execute(policy, log, actor=actor, control=control)
        capabilities: list = [Hooks(before_tool_execute=hook)]
    else:
        capabilities = [RecusalAuditedGate(policy, log=log, actor=actor, control=control)]
    result = scripted_agent(calls, tools=tools, capabilities=capabilities).run_sync("go")
    return responses_of(result), list(EXECUTED)


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
    calls = [
        ("echo", {"text": "hello"}, "c1"),
        ("run_sql", {"sql": "DELETE FROM audit_log"}, "c2"),
        ("run_sql", {"sql": "SELECT * FROM audit_log"}, "c3"),
    ]
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "pydantic-ai-audit.jsonl")
        log = AuditLog(path=path)
        responses, executed = run_calls(calls, log, actor="demo-agent")

        print("RECUSAL - Pydantic AI agent, every call on the record (offline)\n")
        for name, _, call_id in calls:
            print(f"  {call_id} {name:<8} {responses[call_id][0]}")
        print(f"  executed: {', '.join(name for name, _ in executed)}\n")

        # One model turn's calls can complete in any order: read entries by tool_use_id,
        # never by position.
        by_id = {entry["action"]["tool_use_id"]: entry for entry in log.entries}
        for name, _, call_id in calls:
            entry = by_id[call_id]
            action = entry["action"]
            print(
                f"  {call_id} {action['surface']} {action['tool']:<8} "
                f"{action['decision']:<5} {entry['actor']} {entry['hash'][:12]}..."
            )

        code, text = verify_with_cli(path)
        print(f"\n  verify intact:   exit {code}  {text}")

        tamper(path, "c2")  # the refused DELETE now reads as allowed
        code, text = verify_with_cli(path)
        print(f"  verify tampered: exit {code}  {text}")


if __name__ == "__main__":
    main()
