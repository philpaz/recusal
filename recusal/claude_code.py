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

A hand-written policy like the above is a *deny-list* (refuse known-bad, defer the rest).
For high-stakes channels prefer :func:`allowlist_policy` (default-deny: nothing runs
unless affirmatively named), see the "Allowlist mode" section below.

No Anthropic-SDK dependency, this only speaks the hook's stdin/stdout JSON.
"""

import json
import os
import re
import shlex
import sys
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Tuple

from .evidence import Finding, compute_verdict

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
        # strict at the enforcement boundary: ambiguous evidence (a dict with no
        # status/passed) fails closed here rather than silently degrading to PASS.
        verdict = compute_verdict(findings, strict=True)
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
        if "tool_name" not in event:
            raise ValueError("PreToolUse event missing 'tool_name'")
        tool_input = event.get("tool_input", {})
        if not isinstance(tool_input, dict):
            raise ValueError("PreToolUse 'tool_input' is not an object")
        decision, reason = decide(
            event["tool_name"],
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


# --- Allowlist mode (default-deny), the posture for high-stakes channels ----------------
#
# Every deny-list shares one ceiling: it cannot catch a command whose name is built at
# runtime (`c=$'\x72\x6d'; $c -rf /`), and it cannot see code executed *inside* an
# interpreter — `python script.py` is one innocent-looking token followed by a program the
# gate never reads. Allowlist mode inverts the default: **nothing runs unless affirmatively
# named.** Unlisted tools, shell metacharacters (chaining/substitution/expansion), and bare
# interpreters are refused, which closes the write-a-script-then-run-it bypass a deny-list
# cannot. This is the posture behind the strong claim — "the agent could not subvert it" —
# scoped, honestly, to the tool channel routed through the hook.

# Tools that only read; they defer regardless of arguments.
DEFAULT_READ_ONLY_TOOLS: FrozenSet[str] = frozenset({"Read", "Grep", "Glob"})

# First binaries safe regardless of arguments. Mutating tools (git, sed, find, rm) and
# interpreters (python, node, sh, ...) are deliberately absent: an interpreter's argument
# IS a program, so `python script.py` executes code no string check ever vets. Matched as
# the exact first argv token — `./pytest` or `/opt/x/pytest` is NOT the vetted `pytest`.
DEFAULT_SAFE_BINARIES: FrozenSet[str] = frozenset(
    {
        "ls",
        "cat",
        "head",
        "tail",
        "grep",
        "rg",
        "wc",
        "pwd",
        "stat",
        "diff",
        "pytest",
        "ruff",
        "mypy",
    }  # fmt: skip
)

# Presence of any of these means the command can expand into something other than its
# literal argv (chaining, substitution, redirection, escapes) -> refuse, don't reason.
_SHELL_META: FrozenSet[str] = frozenset(";|&`$<>(){}\n\\")

# Interpreter names, recognized only to make the refusal reason precise (an unrecognized
# binary is refused anyway — recognition never widens what is allowed).
_INTERPRETER_NAMES: FrozenSet[str] = frozenset(
    {
        "sh",
        "bash",
        "zsh",
        "dash",
        "ksh",
        "fish",
        "python",
        "perl",
        "ruby",
        "node",
        "php",
        "pwsh",
        "powershell",
        "cmd",
        "deno",
        "bun",
        "lua",
        "osascript",
    }  # fmt: skip
)

_WRITE_TOOLS: FrozenSet[str] = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})


def _is_interpreter(argv0: str) -> bool:
    """True if the first argv token names an interpreter (`python`, `python3.12`,
    `node.exe`, ...). Used only to sharpen the refusal message."""
    base = os.path.basename(argv0.strip().lower())
    if base.endswith(".exe"):
        base = base[: -len(".exe")]
    base = re.sub(r"[\d.]+$", "", base)  # python3.12 -> python
    return base in _INTERPRETER_NAMES


def _under_root(root: str, path: str) -> bool:
    try:
        return os.path.commonpath([root, os.path.abspath(path)]) == root
    except ValueError:  # different drives on Windows
        return False


def _vet_bash(cmd: str, safe_binaries: FrozenSet[str]) -> Optional[str]:
    """Return None if the command is affirmatively safe, else the refusal reason."""
    meta = set(cmd) & _SHELL_META
    if meta:
        pretty = " ".join(sorted(repr(c) for c in meta))
        return f"shell metacharacters ({pretty}) make the command's expansion unvettable"
    try:
        argv = shlex.split(cmd)
    except ValueError:
        return "unbalanced quoting"
    if not argv:
        return "empty command"
    if argv[0] in safe_binaries:
        return None
    if _is_interpreter(argv[0]):
        return (
            f"bare interpreter `{argv[0]}` executes unvetted code (a script file or -c "
            f"program the gate never reads)"
        )
    return f"first binary `{argv[0]}` is not on the allowlist"


def allowlist_policy(
    *,
    safe_binaries: Iterable[str] = DEFAULT_SAFE_BINARIES,
    read_only_tools: Iterable[str] = DEFAULT_READ_ONLY_TOOLS,
    writable_root: Optional[str] = None,
    allow: Optional[Dict[str, Callable[[dict], bool]]] = None,
) -> Policy:
    """Build a default-deny :data:`Policy`: refuse every call not affirmatively vetted.

    - ``read_only_tools`` defer regardless of arguments (default: Read/Grep/Glob).
    - ``Bash`` must have no shell metacharacters and a first binary in ``safe_binaries``.
      Bare interpreters (``python script.py``) are refused: the script is a program the
      gate cannot vet, the exact bypass that defeats a deny-list.
    - Write tools (Write/Edit/MultiEdit/NotebookEdit) are allowed only under
      ``writable_root``; with no root named, all writes are refused.
    - ``allow`` maps a tool name to a predicate ``tool_input -> bool`` and **overrides**
      the built-in vetting for that tool (your predicate is the whole decision).

    Everything else — unlisted tools, MCP tools, anything unnamed — is refused. Plug the
    result into :func:`run_pretooluse_hook`. A vetted call still *defers* to Claude Code's
    normal permission flow; this policy only ever adds refusals.
    """
    safe = frozenset(safe_binaries)
    readonly = frozenset(read_only_tools)
    extra = dict(allow) if allow else {}

    def policy(tool_name: str, tool_input: dict) -> List[Any]:
        def refuse(reason: str) -> List[Any]:
            return [
                Finding.fail(
                    "not_allowlisted",
                    severity="CRITICAL",
                    message=f"`{tool_name}` call is not on the allowlist: {reason}",
                    tool=tool_name,
                )
            ]

        if tool_name in readonly:
            return []
        if tool_name in extra:
            return [] if extra[tool_name](tool_input) else refuse("predicate refused it")
        if tool_name == "Bash":
            reason = _vet_bash(str(tool_input.get("command", "")), safe)
            return [] if reason is None else refuse(reason)
        if tool_name in _WRITE_TOOLS:
            if writable_root is None:
                return refuse("no writable_root is configured, so all writes are refused")
            path = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
            if path and _under_root(os.path.abspath(writable_root), path):
                return []
            return refuse(f"path {path!r} is outside the writable root")
        return refuse("tool is not named in the allowlist")

    return policy
