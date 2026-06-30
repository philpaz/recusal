"""
Claude Code adapter, run Recusal as a PreToolUse hook.

Claude Code fires a ``PreToolUse`` hook before it executes any tool. The hook reads
a JSON event on stdin and writes a decision on stdout. This adapter turns a Recusal
policy, a function ``(tool_name, tool_input) -> findings``, into that decision.

Design: a governance gate should only ever **deny**, never force-allow. So:

    verdict PASS   → DEFER  (emit nothing; Claude Code's normal permission flow runs)
    verdict RETRY  → deny   (with the reasons, so Claude re-plans)
    verdict FAIL   → deny   (a PreToolUse "deny" is honored even under bypassPermissions)

Deferring on PASS is deliberate: the gate adds refusals, it does not strip away
Claude Code's own permission prompts. (Pass ``allow_on_pass=True`` only if you truly
want the gate to auto-approve and bypass the prompt.)

Wire it up in ``.claude/settings.json``::

    {
      "hooks": {
        "PreToolUse": [
          { "matcher": ".*", "hooks": [
            { "type": "command", "command": "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/my_gate.py" }
          ]}
        ]
      }
    }

``my_gate.py``::

    from recusal import Finding
    from recusal.claude_code import run_pretooluse_hook

    def policy(tool_name, tool_input):
        if tool_name == "Bash" and "rm -rf" in tool_input.get("command", ""):
            return [Finding.fail("destructive_bash", severity="CRITICAL",
                                 message="refusing rm -rf")]
        return []   # no opinion → defer to Claude Code

    run_pretooluse_hook(policy)

No Anthropic-SDK dependency, this only speaks the hook's stdin/stdout JSON.
"""

import json
import sys
from typing import Any, Callable, List, Optional, Tuple

from .evidence import compute_verdict

# A policy maps a proposed tool call to evidence findings.
Policy = Callable[[str, dict], List[Any]]


def decide(
    tool_name: str,
    tool_input: dict,
    policy: Policy,
    *,
    allow_on_pass: bool = False,
    fail_closed: bool = True,
) -> Tuple[str, str]:
    """Pure decision: run the policy, fold to a verdict, return ``(decision, reason)``.

    ``decision`` is ``"defer"`` (PASS, and not auto-allowing), ``"allow"`` (PASS with
    ``allow_on_pass=True``), or ``"deny"`` (RETRY/FAIL).
    """
    try:
        findings = policy(tool_name, tool_input) or []
        verdict = compute_verdict(findings)
    except Exception as exc:  # noqa: BLE001, a buggy policy must not silently disable the gate
        if fail_closed:
            return "deny", f"Recusal failed closed (policy error): {exc}"
        return ("allow" if allow_on_pass else "defer"), f"policy error ignored: {exc}"
    if verdict.passed:
        return ("allow" if allow_on_pass else "defer"), verdict.message
    return "deny", (
        f"Recusal refused `{tool_name}` [{verdict.decision.value}]: {verdict.reasons()}"
    )


def run_pretooluse_hook(
    policy: Policy,
    *,
    allow_on_pass: bool = False,
    fail_closed: bool = True,
    stdin: Any = None,
    stdout: Any = None,
) -> Optional[dict]:
    """Read a Claude Code PreToolUse event on stdin, apply ``policy``, emit the decision.

    On a deny (or an explicit allow), writes the PreToolUse ``hookSpecificOutput`` JSON
    and returns it. On a defer, writes nothing and returns ``None``, Claude Code then
    proceeds with its normal permission flow.

    A malformed envelope (unparseable stdin, or valid JSON that is not an object) is
    treated like a policy error: it **fails closed** to a ``deny`` by default, so a
    garbled or truncated event cannot silently skip the gate. Pass ``fail_closed=False``
    to defer instead.
    """
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    try:
        event = json.load(stdin)
        if not isinstance(event, dict):
            raise ValueError("PreToolUse event is not a JSON object")
        tool_input = event.get("tool_input", {})
        if not isinstance(tool_input, dict):
            tool_input = {}
        decision, reason = decide(
            event.get("tool_name", ""),
            tool_input,
            policy,
            allow_on_pass=allow_on_pass,
            fail_closed=fail_closed,
        )
    except Exception as exc:  # noqa: BLE001 - a malformed event must not silently disable the gate
        if not fail_closed:
            return None  # fail-open: defer to Claude Code's normal flow
        decision, reason = "deny", f"Recusal failed closed: malformed PreToolUse event ({exc})"

    if decision == "defer":
        return None

    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,  # "allow" | "deny"
            "permissionDecisionReason": reason,
        }
    }
    json.dump(output, stdout)
    stdout.write("\n")
    return output
