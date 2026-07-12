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

Wire it up in ``.claude/settings.json``. Use the interpreter-probe launcher below rather
than a bare ``python3 ...``. Exit-code semantics, exactly: a deny is exit 0 WITH
``permissionDecision: "deny"`` JSON (honored as a block); a defer is exit 0 with no
output; exit 2 is Claude's blocking-failure signal; any OTHER nonzero exit is a
non-blocking error and the tool call proceeds. That last rule means a hook whose command
fails to *launch* silently disables the gate, so a bare ``python3`` fails open on any
host without a ``python3`` on PATH (Windows, and any box where the py3 is ``python``).
The loop runs the first ``python3``/``python``/``py`` that is >=3.9 and coerces ANY
nonzero exit -- missing interpreter, wrong version, or a hook that fails to run -- into
``exit 2``, so those gate-process failure modes fail **closed**.

**Windows:** shell-form hooks run under Git Bash when it is installed, and Claude Code
falls back to PowerShell when it is not - where this POSIX loop is a *parse error* with a
non-blocking exit code, i.e. the gate silently disables. ``python -m recusal init``
therefore registers a PowerShell-native launcher (explicit ``"shell": "powershell"``) on
Windows; the POSIX form below is for macOS/Linux and Windows-with-Git-Bash only::

    {
      "hooks": {
        "PreToolUse": [
          { "matcher": ".*", "hooks": [
            { "type": "command", "command": "for p in python3 python py; do \"$p\" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null && { \"$p\" \"$CLAUDE_PROJECT_DIR/.claude/hooks/my_gate.py\"; rc=$?; [ \"$rc\" = 0 ] || { echo 'gate: hook did not run cleanly; failing closed' >&2; exit 2; }; exit 0; }; done; echo 'gate: no python>=3.9; failing closed' >&2; exit 2" }
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

import hashlib
import json
import os
import re
import shlex
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Tuple

from .evidence import Finding, Verdict, compute_verdict

if TYPE_CHECKING:  # runtime never needs the import; the hook only calls audit.append
    from .audit import AuditLog

# A policy maps a proposed tool call to evidence findings.
Policy = Callable[[str, dict], List[Any]]


def _adjudicate(
    tool_name: str,
    tool_input: dict,
    policy: Policy,
    *,
    allow_on_pass: bool,
    fail_closed: bool,
) -> Tuple[str, str, Verdict]:
    """``decide`` plus the Verdict that drove it, so a caller can put it on the record.

    A policy error has no verdict of its own; one is synthesized so the audit entry
    states what actually happened (CRITICAL when it denied, WARNING when ignored)."""
    try:
        findings = policy(tool_name, tool_input) or []
        # strict at the enforcement boundary: ambiguous evidence (a dict with no
        # status/passed) fails closed here rather than silently degrading to PASS.
        verdict = compute_verdict(findings, strict=True)
    except Exception as exc:  # noqa: BLE001, a buggy policy must not silently disable the gate
        if fail_closed:
            verdict = compute_verdict(
                [Finding.fail("recusal_policy_error", severity="CRITICAL", message=str(exc))]
            )
            return "deny", f"Recusal failed closed (policy error): {exc}", verdict
        verdict = compute_verdict(
            [
                Finding.fail(
                    "recusal_policy_error",
                    severity="WARNING",
                    message=f"ignored (fail_closed=False): {exc}",
                )
            ]
        )
        return ("allow" if allow_on_pass else "defer"), f"policy error ignored: {exc}", verdict
    if verdict.passed:
        return ("allow" if allow_on_pass else "defer"), verdict.message, verdict
    return (
        "deny",
        f"Recusal refused `{tool_name}` [{verdict.decision.value}]: {verdict.reasons()}",
        verdict,
    )


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
    decision, reason, _ = _adjudicate(
        tool_name, tool_input, policy, allow_on_pass=allow_on_pass, fail_closed=fail_closed
    )
    return decision, reason


#: Control-identity fields the IMPLEMENTATION owns. Caller-supplied values for these are
#: discarded, never merged: an audit record whose "recusal_version" or "manifest_sha256"
#: could be written by the caller would be provenance theater, not provenance.
_AUTHORITATIVE_CONTROL_KEYS = frozenset({"recusal_version", "manifest_sha256"})


def _control_identity(policy: Policy, control: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The control identity recorded on EVERY audit path, built in one place.

    Caller metadata first (reserved keys stripped), then the implementation-owned truth
    written over it: the actual package version always, and the manifest digest a
    :func:`recusal.mcp.manifest_policy` verified during THIS adjudication when present.

    The digest is read through the policy's ``get_control_identity()`` when it has one:
    :func:`recusal.mcp.manifest_policy` keeps it in a ``ContextVar``, so under
    concurrent reuse of one policy object each audit record sees the digest its own
    invocation verified, set only after successful validation (a corrupt manifest or a
    non-MCP call never carries manifest provenance). A plain ``last_manifest_digest``
    attribute is honored as a fallback for custom policy objects; being shared mutable
    state, that seam is safe for sequential use only.
    """
    from . import __version__ as _recusal_version  # local import: no cycle at load

    identity = {k: v for k, v in (control or {}).items() if k not in _AUTHORITATIVE_CONTROL_KEYS}
    identity["recusal_version"] = _recusal_version
    getter = getattr(policy, "get_control_identity", None)
    if callable(getter):
        try:
            manifest_digest = dict(getter()).get("manifest_sha256")
        except Exception:  # noqa: BLE001, a broken getter must not take down the audit path
            manifest_digest = None
    else:
        manifest_digest = getattr(policy, "last_manifest_digest", None)
    if isinstance(manifest_digest, str):
        identity["manifest_sha256"] = manifest_digest
    return identity


def _input_fingerprint(tool_input: dict) -> str:
    """SHA-256 over the canonical JSON of the proposed tool input. The audit entry binds
    to the exact proposed call without embedding its contents (a Write's file body, an
    env value): hashes only, the same doctrine as the MCP manifest."""
    canonical = json.dumps(
        tool_input, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_pretooluse_hook(
    policy: Policy,
    *,
    allow_on_pass: bool = False,
    fail_closed: bool = True,
    audit: Optional["AuditLog"] = None,
    actor: Optional[str] = None,
    control: Optional[Dict[str, Any]] = None,
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

    Pass ``audit=`` (a :class:`recusal.audit.AuditLog`) to put every adjudication on the
    record - defer, allow, and deny alike - as one hash-chained entry naming the tool,
    the decision, the reasons, and a SHA-256 fingerprint of the proposed ``tool_input``
    (contents are never embedded). ``actor`` labels the entries; when omitted, the
    event's ``session_id`` is used if present. If the log cannot be written the hook
    fails **closed** to a deny - the record is part of the control - unless
    ``fail_closed=False``. The hook runs as a fresh process per tool call, so open the
    log with ``AuditLog(path, resume="tail")`` to avoid re-reading a grown log each call.

    ``control`` names the CONTROL IDENTITY on each audit entry: a verdict is replayable
    only when the adjudication rules are identifiable, so "same evidence" is
    insufficient if the policy changed. The recusal package version is recorded
    automatically; pass ``{"policy_id": ..., "policy_version": ...}`` (any JSON-safe
    identity) for the policy; a :func:`recusal.mcp.manifest_policy` contributes the
    manifest content digest it enforced automatically.
    """
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    tool_name: Optional[str] = None
    input_sha256: Optional[str] = None
    event_ids: Dict[str, Any] = {}
    try:
        event = json.load(stdin)
        if not isinstance(event, dict):
            raise ValueError("PreToolUse event is not a JSON object")
        if "tool_name" not in event:
            raise ValueError("PreToolUse event missing 'tool_name'")
        if not isinstance(event["tool_name"], str) or not event["tool_name"]:
            # a null/empty/non-string tool name is a malformed envelope, not something a
            # policy should be asked to reason about (fail-closed posture, same as above)
            raise ValueError("PreToolUse 'tool_name' must be a nonempty string")
        tool_input = event.get("tool_input", {})
        if not isinstance(tool_input, dict):
            raise ValueError("PreToolUse 'tool_input' is not an object")
        tool_name = event["tool_name"]
        input_sha256 = _input_fingerprint(tool_input)
        # transcript linkage for the audit record: prompt_id is in the documented event
        # shape; tool_use_id is recorded defensively should the event ever carry one.
        for key in ("prompt_id", "tool_use_id"):
            if isinstance(event.get(key), str) and event[key]:
                event_ids[key] = event[key]
        if actor is None and isinstance(event.get("session_id"), str):
            actor = event["session_id"]
        decision, reason, verdict = _adjudicate(
            event["tool_name"],
            tool_input,
            policy,
            allow_on_pass=allow_on_pass,
            fail_closed=fail_closed,
        )
    except Exception as exc:  # noqa: BLE001 - a malformed event must not silently disable the gate
        if not fail_closed:
            # fail-open: defer to Claude Code's normal flow. Still a decision: if a log
            # was asked for, record it (best-effort; PASS with the warning on record).
            if audit is not None:
                verdict = compute_verdict(
                    [
                        Finding.fail(
                            "recusal_malformed_event",
                            severity="WARNING",
                            message=f"deferred (fail_closed=False): {exc}",
                        )
                    ]
                )
                try:
                    audit.append(
                        verdict,
                        action={
                            "surface": "claude_code.pretooluse",
                            "tool": tool_name,
                            "decision": "defer",
                            "reason": f"malformed PreToolUse event ignored: {exc}",
                            # the SAME control-identity construction as every other
                            # audit path: a fail-open record still names what decided
                            "control": _control_identity(policy, control),
                        },
                        actor=actor,
                    )
                except Exception:  # noqa: BLE001 - fail-open mode was chosen explicitly
                    pass
            return None
        decision, reason = "deny", f"Recusal failed closed: malformed PreToolUse event ({exc})"
        verdict = compute_verdict(
            [Finding.fail("recusal_malformed_event", severity="CRITICAL", message=str(exc))]
        )

    if audit is not None:
        action: Dict[str, Any] = {
            "surface": "claude_code.pretooluse",
            "tool": tool_name,
            "decision": decision,
            "reason": reason,
        }
        if input_sha256 is not None:
            action["input_sha256"] = input_sha256
        action.update(event_ids)
        action["control"] = _control_identity(policy, control)
        try:
            audit.append(verdict, action=action, actor=actor)
        except Exception as exc:  # noqa: BLE001 - an unwritable log must not go unnoticed
            if fail_closed:
                decision, reason = (
                    "deny",
                    f"Recusal failed closed: audit log unavailable ({exc}); "
                    "the record is part of the control",
                )

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
# interpreter, `python script.py` is one innocent-looking token followed by a program the
# gate never reads. Allowlist mode inverts the default: **nothing runs unless affirmatively
# named.** Unlisted tools, shell metacharacters (chaining/substitution/redirection), and bare
# interpreters are refused, which closes the write-a-script-then-run-it bypass a deny-list
# cannot. The claim, stated precisely: within a correctly registered routed tool channel,
# an unapproved capability is refused by default rather than inferred safe. Read-only means
# NONMUTATING, not authorized for all data - `cat` reads a credential file as happily as a
# README - so add path/subject-level read rules where confidentiality matters.

# Tools that only read; they defer regardless of arguments.
DEFAULT_READ_ONLY_TOOLS: FrozenSet[str] = frozenset({"Read", "Grep", "Glob"})

# First binaries safe *regardless of their arguments* -- read/inspect tools with no flag
# that spawns a process, imports code, or writes a file. Mutating tools (git, sed, find, rm)
# and interpreters (python, node, sh, ...) are deliberately absent: an interpreter's argument
# IS a program, so `python script.py` executes code no string check ever vets. Matched as
# the exact first argv token -- `./cat` or `/opt/x/cat` is NOT the vetted `cat`.
#
# NOT here, and why -- these run arbitrary code THROUGH an argument, so allowlisting the
# binary name would reopen the exact bypass this mode exists to close:
#   pytest  -> auto-imports conftest.py from the working tree
#   mypy    -> imports `[tool.mypy] plugins` from pyproject.toml
#   rg      -> `--pre <cmd>` / `--hostname-bin <cmd>` spawn an arbitrary binary
#   git     -> `-c core.pager=`, `-c alias.x='!sh'`, `-c core.sshCommand=` all execute
# If you add a binary, it must be safe under EVERY argument the agent could pass; a tool
# that reads a config/plugin/hook or has an exec flag does not qualify. Gate those with an
# `allow=` predicate that vets the arguments instead.
DEFAULT_SAFE_BINARIES: FrozenSet[str] = frozenset(
    {
        "ls",
        "cat",
        "head",
        "tail",
        "grep",
        "wc",
        "pwd",
        "stat",
        "diff",
    }  # fmt: skip
)

# Presence of any of these means the command can chain, substitute, redirect, or escape
# its way past the single-argv reading we vet -> refuse, don't reason. Glob (`*`, `?`,
# `[`) and tilde expansion are deliberately NOT in the set: they only widen which paths
# an allowlisted read-only binary reads, and any literal path is equally readable by
# design (`cat /etc/passwd` vets the same as `cat *`), so refusing them would add
# friction without removing a capability. They cannot smuggle in a second binary: an
# argv[0] containing a glob or tilde is not a literal allowlist match and is refused.
_SHELL_META: FrozenSet[str] = frozenset(";|&`$<>(){}\n\\")

# Interpreter names, recognized only to make the refusal reason precise (an unrecognized
# binary is refused anyway, recognition never widens what is allowed).
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
    # realpath both sides so a symlink INSIDE the writable root that points outside it does
    # not escape confinement (commonpath alone would compare the innocent-looking link path).
    try:
        root_real = os.path.realpath(root)
        return os.path.commonpath([root_real, os.path.realpath(path)]) == root_real
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
      ``writable_root``; with no root named, all writes are refused. ``writable_root`` is
      resolved against the process working directory, so prefer an **absolute** path -- a
      relative root like ``./workspace`` shifts if the hook is ever launched from a different
      CWD (it fails toward refusal, never toward a wider root, but absolute is unambiguous).
    - ``allow`` maps a tool name to a predicate ``tool_input -> bool`` and **overrides**
      the built-in vetting for that tool (your predicate is the whole decision).

    Everything else, unlisted tools, MCP tools, anything unnamed, is refused. Plug the
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
