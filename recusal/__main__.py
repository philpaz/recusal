"""The ``recusal`` command line: scaffold the gate, adjudicate in CI.

Subcommands:

- ``init``: scaffold a fail-closed Claude Code PreToolUse gate (detailed below);
- ``verdict``: adjudicate a findings JSON file into PASS / RETRY / FAIL with blocking
  exit codes (0 / 1 / 2), the CI primitive - any tool can emit findings, recusal
  adjudicates them, and a nonzero exit blocks the merge;
- ``audit verify``: check a hash-chained audit log's integrity (``recusal.audit``);
- ``doctor``: health-check a scaffolded gate, so "the gate silently isn't installed"
  is caught by CI instead of discovered during an incident;
- ``mcp pin`` / ``mcp verify``: pin an MCP server tool catalog to a deterministic
  manifest, then refuse drift (``recusal.mcp``), the discovery-boundary counterpart
  of the call-time gate.

The CI commands share one discipline: an operational error (unreadable file, invalid
JSON, malformed anchor) exits 2, indistinguishable from FAIL on purpose. A gate that
cannot adjudicate must refuse, not wave the job through.

``init``, in detail. The README's manual path (hand-editing ``.claude/settings.json`` and writing a gate
script) is deliberate but high-friction, and a JSON typo in a hook command *silently*
disables governance. This scaffolder emits exactly the pattern this repository dogfoods:

- ``.claude/hooks/recusal_gate.py``, a thin shim wiring a shipped policy
  (:func:`recusal.deny_list.deny_list_policy` or
  :func:`recusal.claude_code.allowlist_policy`) into
  :func:`recusal.claude_code.run_pretooluse_hook`;
- a ``PreToolUse`` entry in ``.claude/settings.json`` whose launcher probes for a
  python3/python/py that is >=3.9 and coerces ANY nonzero exit, missing interpreter,
  wrong version, unimportable ``recusal``, into exit 2, the one *blocking* hook exit
  code, so the gate fails CLOSED rather than open.

Fail-safe file semantics, pinned by tests:

- an existing gate file is NEVER overwritten (it is the user's policy);
- an existing ``settings.json`` is merged, never clobbered, and if it does not parse
  it is left byte-for-byte untouched and the manual snippet is printed instead;
- running twice is a no-op.

This module is a convenience around the kernel, not part of it: no model, no network,
standard library only, and nothing here participates in verdict computation.
"""

import argparse
import json
import os
import re
import shutil
import sys
from typing import Any, Dict, List, NamedTuple, Optional, TextIO, Tuple

from . import __version__
from .audit import GENESIS
from .audit import load as _load_audit_entries
from .audit import verify_file as _verify_audit_file
from .evidence import Decision, Finding, Verdict, compute_verdict
from .mcp import (
    McpObservation,
    build_manifest,
    diff_observation,
    diff_source,
    load_manifest,
    manifest_to_text,
    screen_server_instructions,
    screen_tool_declarations,
)
from .mcp_fetch import (
    McpFetchError,
    fetch_server_stdio,
    servers_from_claude_config,
    split_command,
)

#: The fail-closed POSIX launcher, verbatim the command this repository registers for
#: itself (see ``.claude/settings.json.example``). A deny is exit 0 with deny JSON and a
#: defer is exit 0 with no output; exit 2 is Claude's BLOCKING failure signal, and any
#: OTHER nonzero exit (a command that cannot launch, a crashed hook) is a NON-blocking
#: error that lets the tool call proceed. This loop coerces every nonzero gate-process
#: failure into exit 2 so a broken or absent interpreter refuses the tool call instead
#: of waving it through.
#:
#: Shell reality on Windows: hook commands run under Git Bash when it is installed, and
#: Claude Code FALLS BACK TO POWERSHELL when it is not - where this POSIX loop is a parse
#: error, exit 1, a NON-blocking code, i.e. the gate silently disables (live-verified).
#: That is why ``init`` registers the PowerShell launcher below, with an explicit
#: ``"shell": "powershell"``, on Windows.
LAUNCHER_COMMAND_POSIX = (
    'for p in python3 python py; do "$p" -c \'import sys; sys.exit(0 if sys.version_info'
    ' >= (3, 9) else 1)\' 2>/dev/null && { "$p"'
    ' "$CLAUDE_PROJECT_DIR/.claude/hooks/recusal_gate.py"; rc=$?; [ "$rc" = 0 ] ||'
    " { echo 'recusal gate: hook did not run cleanly; failing closed' >&2; exit 2; };"
    " exit 0; }; done; echo 'recusal gate: no working python>=3.9 interpreter; failing"
    " closed' >&2; exit 2"
)

#: Back-compat alias (the POSIX form was previously the only launcher).
LAUNCHER_COMMAND = LAUNCHER_COMMAND_POSIX

#: The same semantics in PowerShell, for Windows hosts: probe ``py``/``python``/
#: ``python3`` for >=3.9, run the gate, and coerce EVERY failure (no interpreter, wrong
#: version, gate crash) into exit 2, the one blocking hook code. PowerShell is always
#: present on Windows, so this launcher does not depend on Git Bash being installed.
#: Registered with an explicit ``"shell": "powershell"`` so Git Bash, when present, never
#: tries (and fails) to parse it.
LAUNCHER_COMMAND_POWERSHELL = (
    "$ErrorActionPreference = 'Continue'; foreach ($p in @('py', 'python', 'python3')) {"
    " if (-not (Get-Command $p -ErrorAction SilentlyContinue)) { continue };"
    " & $p -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' *> $null;"
    " if ($LASTEXITCODE -ne 0) { continue };"
    ' & $p "$env:CLAUDE_PROJECT_DIR/.claude/hooks/recusal_gate.py";'
    " if ($LASTEXITCODE -eq 0) { exit 0 };"
    " [Console]::Error.WriteLine('recusal gate: hook did not run cleanly; failing closed');"
    " exit 2 };"
    " [Console]::Error.WriteLine("
    "'recusal gate: no working python>=3.9 interpreter; failing closed'); exit 2"
)

_WINDOWS = os.name == "nt"


def _hook_entries(launcher: str = "auto") -> List[dict]:
    """The PreToolUse entries to register for ``launcher``.

    ``auto`` picks by host: PowerShell on Windows (present with or without Git Bash),
    POSIX elsewhere. ``both`` registers the two of them for a settings.json shared
    across operating systems - on any host at least one launcher is functional and
    blocking, and a deny from either blocks the call (the caveat: on Windows WITH Git
    Bash both run, so the gate adjudicates twice per call).
    """
    if launcher == "auto":
        launcher = "powershell" if _WINDOWS else "posix"
    entries: List[dict] = []
    if launcher in ("posix", "both"):
        entries.append(
            {"matcher": ".*", "hooks": [{"type": "command", "command": LAUNCHER_COMMAND_POSIX}]}
        )
    if launcher in ("powershell", "both"):
        entries.append(
            {
                "matcher": ".*",
                "hooks": [
                    {
                        "type": "command",
                        "command": LAUNCHER_COMMAND_POWERSHELL,
                        # load-bearing: without it, Git Bash (when installed) would try
                        # to parse PowerShell and fail with a NON-blocking exit code.
                        "shell": "powershell",
                    }
                ],
            }
        )
    if not entries:
        raise ValueError(f"unknown launcher: {launcher!r}")
    return entries


#: The marker that makes a re-run idempotent: a PreToolUse hook whose command mentions
#: this filename is recognized as an already-installed recusal gate.
GATE_FILENAME = "recusal_gate.py"

_GATE_HEADER = '''\
#!/usr/bin/env python3
"""Recusal PreToolUse gate, scaffolded by ``python -m recusal init``.

Every Claude Code tool call in this project is adjudicated here BEFORE it runs; a deny
holds even under bypassPermissions. If ``recusal`` is not importable (not installed in
the interpreter the launcher found), this script exits nonzero and the registered
launcher fails CLOSED, refusing the tool call rather than skipping the gate.

The policy below is a starting point, not a guarantee; edit it, it is yours.
Recipes: https://github.com/philpaz/recusal/blob/main/docs/COOKBOOK.md
Threat model and the deny-list ceiling: SECURITY.md in the same repo.
"""
'''

_GATE_DENY_LIST = (
    _GATE_HEADER
    + """
from recusal.claude_code import run_pretooluse_hook
from recusal.deny_list import deny_list_policy

# Reference deny-list: refuse known-destructive commands (recursive deletes,
# force-pushes, secret-file writes, edits to this gate's own configuration),
# DEFER everything else to Claude Code's normal permission flow. A deny-list is
# a baseline for broad channels, a literal matcher can be obfuscated past; for a
# narrow high-stakes channel, re-run init with `--posture allowlist`.
policy = deny_list_policy()

if __name__ == "__main__":
    run_pretooluse_hook(policy)
"""
)

_GATE_ALLOWLIST = (
    _GATE_HEADER
    + """
from recusal.claude_code import allowlist_policy, run_pretooluse_hook

# Default-deny allowlist: NOTHING runs unless affirmatively named. Unlisted tools,
# shell metacharacters, and bare interpreters are refused, which closes the
# write-a-script-then-run-it bypass by construction. The trade is friction: you
# enumerate the capability set, and the gate fails toward refusal until you do.
policy = allowlist_policy(writable_root={writable_root!r})

if __name__ == "__main__":
    run_pretooluse_hook(policy)
"""
)


def gate_source(posture: str, writable_root: str = "./workspace") -> str:
    """Return the gate-script source for ``posture`` (``deny-list`` | ``allowlist``)."""
    if posture == "deny-list":
        return _GATE_DENY_LIST
    if posture == "allowlist":
        return _GATE_ALLOWLIST.format(writable_root=writable_root)
    raise ValueError(f"unknown posture: {posture!r}")


def _recusal_hooks(settings: dict) -> List[dict]:
    """Every PreToolUse hook dict in ``settings`` that references the recusal gate."""
    found: List[dict] = []
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        return found
    groups = hooks.get("PreToolUse", [])
    if not isinstance(groups, list):
        return found
    for group in groups:
        if not isinstance(group, dict):
            continue
        for hook in group.get("hooks", []) or []:
            if isinstance(hook, dict) and GATE_FILENAME in str(hook.get("command", "")):
                found.append(hook)
    return found


def _recusal_hook_commands(settings: dict) -> List[str]:
    """Every PreToolUse command in ``settings`` that references the recusal gate."""
    return [str(hook["command"]) for hook in _recusal_hooks(settings)]


def _has_recusal_hook(settings: dict) -> bool:
    """True if any PreToolUse command already references the recusal gate."""
    return bool(_recusal_hook_commands(settings))


def merge_settings(
    existing_text: Optional[str], *, launcher: str = "auto"
) -> Tuple[Optional[str], str]:
    """Merge the recusal PreToolUse entry into a settings.json body.

    Returns ``(new_text, status)`` where status is one of ``created`` / ``merged`` /
    ``already-installed`` / ``unparseable`` / ``unexpected-shape``. ``new_text`` is
    ``None`` whenever the file must not be written (idempotent no-op or refusal),
    so a caller cannot accidentally clobber a file this function refused to merge.
    """
    entries = _hook_entries(launcher)
    if existing_text is None:
        body = {"hooks": {"PreToolUse": entries}}
        return json.dumps(body, indent=2) + "\n", "created"

    try:
        settings = json.loads(existing_text)
    except ValueError:
        return None, "unparseable"
    if not isinstance(settings, dict):
        return None, "unexpected-shape"
    if _has_recusal_hook(settings):
        return None, "already-installed"

    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        return None, "unexpected-shape"
    pre = hooks.setdefault("PreToolUse", [])
    if not isinstance(pre, list):
        return None, "unexpected-shape"
    pre.extend(entries)
    return json.dumps(settings, indent=2) + "\n", "merged"


def _manual_snippet(launcher: str = "auto") -> str:
    entry = {"hooks": {"PreToolUse": _hook_entries(launcher)}}
    return json.dumps(entry, indent=2)


def repair_launcher(project_dir: str, launcher: str = "auto", stdout=None) -> int:
    """Replace the CANONICAL recusal launcher(s) in settings.json with the right ones
    for this host, changing nothing else. Returns a process exit code.

    The migration path for pre-0.4.2 Windows installs: ``init`` treats any existing
    recusal hook as already-installed and changes nothing, so a host whose registered
    POSIX launcher would fail OPEN under the PowerShell fallback had no automated way
    forward. This recognizes the exact canonical launchers (never a custom command),
    removes them, registers the host-appropriate entries, preserves every other hook
    and setting byte-for-byte in spirit, never touches the gate policy file, prints the
    exact change, and is a no-op the second time.
    """
    out = stdout if stdout is not None else sys.stdout
    settings_path = os.path.join(project_dir, ".claude", "settings.json")
    if not os.path.exists(settings_path):
        out.write(f"{settings_path} does not exist; run `recusal init` for a fresh install\n")
        return 1
    with open(settings_path, encoding="utf-8") as f:
        existing_text = f.read()
    try:
        settings = json.loads(existing_text)
    except ValueError:
        out.write(f"REFUSING to edit {settings_path} (unparseable); fix it by hand\n")
        return 1
    if not isinstance(settings, dict):
        out.write(f"REFUSING to edit {settings_path} (unexpected shape)\n")
        return 1
    canonical = {LAUNCHER_COMMAND_POSIX, LAUNCHER_COMMAND_POWERSHELL}
    hooks = settings.get("hooks")
    groups = hooks.get("PreToolUse") if isinstance(hooks, dict) else None
    if not isinstance(groups, list):
        out.write("no PreToolUse hooks registered; run `recusal init` for a fresh install\n")
        return 1
    removed = 0
    custom = 0
    kept_groups = []
    for group in groups:
        if not isinstance(group, dict):
            kept_groups.append(group)
            continue
        kept_hooks = []
        for hook in group.get("hooks", []) or []:
            command = str(hook.get("command", "")) if isinstance(hook, dict) else ""
            if command in canonical:
                removed += 1
                continue  # the known launcher, ours to replace
            if GATE_FILENAME in command:
                custom += 1  # a customized recusal hook is the USER'S; never touched
            kept_hooks.append(hook)
        if kept_hooks or not group.get("hooks"):
            kept_groups.append({**group, "hooks": kept_hooks} if "hooks" in group else group)
    entries = _hook_entries(launcher)
    settings.setdefault("hooks", {})["PreToolUse"] = kept_groups + entries
    new_text = json.dumps(settings, indent=2) + "\n"
    if new_text == existing_text:
        out.write("launcher already matches this host; no change\n")
        return 0
    with open(settings_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(new_text)
    launchers = ", ".join(
        "powershell" if h["hooks"][0].get("shell") == "powershell" else "posix" for h in entries
    )
    out.write(
        f"replaced {removed} canonical launcher(s) with {launchers} launcher(s) in "
        f"{settings_path}\n"
    )
    if custom:
        out.write(
            f"left {custom} customized recusal hook(s) untouched; review them against the "
            "canonical launchers yourself\n"
        )
    return 0


def init(
    project_dir: str,
    posture: str = "deny-list",
    writable_root: str = "./workspace",
    launcher: str = "auto",
    stdout=None,
) -> int:
    """Scaffold the gate into ``project_dir``. Returns a process exit code."""
    out = stdout if stdout is not None else sys.stdout
    claude_dir = os.path.join(project_dir, ".claude")
    hooks_dir = os.path.join(claude_dir, "hooks")
    gate_path = os.path.join(hooks_dir, GATE_FILENAME)
    settings_path = os.path.join(claude_dir, "settings.json")

    # 1) The gate script. Never overwrite: an existing gate is the user's policy.
    if os.path.exists(gate_path):
        out.write(f"gate exists, left untouched: {gate_path}\n")
    else:
        os.makedirs(hooks_dir, exist_ok=True)
        with open(gate_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(gate_source(posture, writable_root))
        out.write(f"wrote {gate_path}  (posture: {posture})\n")

    # 2) settings.json: create or merge, never clobber.
    existing = None
    if os.path.exists(settings_path):
        with open(settings_path, encoding="utf-8") as f:
            existing = f.read()
    new_text, status = merge_settings(existing, launcher=launcher)

    if status in ("unparseable", "unexpected-shape"):
        out.write(
            f"REFUSING to edit {settings_path} ({status}); it was left untouched.\n"
            "Add this PreToolUse entry by hand:\n" + _manual_snippet(launcher) + "\n"
        )
        return 1
    if status == "already-installed":
        out.write(f"settings already register the gate, no change: {settings_path}\n")
    else:
        assert new_text is not None  # created/merged always carry a body
        with open(settings_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(new_text)
        out.write(f"{status} {settings_path}\n")

    out.write(
        "\nDone. Start a Claude Code session in this project; Claude Code will ask you\n"
        "to confirm the new hook (a permission-changing hook is a deliberate step).\n"
        "The gate refuses destructive calls before they run, even under bypassPermissions.\n"
        "Edit your policy in .claude/hooks/recusal_gate.py; recipes: docs/COOKBOOK.md.\n"
    )
    return 0


# --- CI adjudication commands ------------------------------------------------------------
#
# ``verdict`` / ``audit verify`` / ``doctor`` expose the kernel to CI with blocking exit
# codes: PASS -> 0, RETRY -> 1, FAIL -> 2. An operational error (unreadable file, invalid
# JSON, malformed anchor) exits 2, indistinguishable from FAIL on purpose: a gate that
# cannot adjudicate must refuse, not wave the job through.

EXIT_BY_DECISION = {Decision.PASS: 0, Decision.RETRY: 1, Decision.FAIL: 2}
EXIT_ERROR = 2


def _finding_brief(finding: Finding) -> Dict[str, Any]:
    return {
        "check": finding.check,
        "severity": finding.severity.value,
        "passed": finding.passed,
        "message": finding.message,
    }


def _verdict_payload(verdict: Verdict) -> Dict[str, Any]:
    """The stable JSON shape ``--json`` emits (field names match the audit log's summary)."""
    return {
        "decision": verdict.decision.value,
        "highest_severity": verdict.highest_severity.value,
        "message": verdict.message,
        "reasons": verdict.reasons(),
        "failures": [_finding_brief(f) for f in verdict.failures],
        "warnings": [_finding_brief(f) for f in verdict.warnings],
        "metrics": len(verdict.metrics),
        "exit_code": EXIT_BY_DECISION[verdict.decision],
    }


def _emit_verdict(verdict: Verdict, as_json: bool, out: TextIO) -> int:
    if as_json:
        out.write(json.dumps(_verdict_payload(verdict), indent=2, sort_keys=True) + "\n")
    else:
        out.write(f"{verdict.decision.value} - {verdict.message}\n")
        for f in verdict.failures:
            out.write(f"  FAILED {f.check} [{f.severity.value}]: {f.message}\n")
        for f in verdict.warnings:
            out.write(f"  warning {f.check}: {f.message}\n")
    return EXIT_BY_DECISION[verdict.decision]


def _fail_closed(reason: str, as_json: bool, out: TextIO) -> int:
    """Emit an operational failure as a refusal and return the blocking exit code."""
    if as_json:
        payload = {
            "decision": Decision.FAIL.value,
            "failed_closed": True,
            "message": reason,
            "exit_code": EXIT_ERROR,
        }
        out.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        out.write(f"FAIL (failed closed) - {reason}\n")
    return EXIT_ERROR


def verdict_command(
    findings_path: str,
    *,
    as_json: bool = False,
    lenient: bool = False,
    stdout: Optional[TextIO] = None,
    stdin: Optional[TextIO] = None,
) -> int:
    """Adjudicate a findings JSON file (or ``-`` for stdin). Exit: PASS 0, RETRY 1, FAIL 2.

    Accepts a JSON array of findings, or an object with a ``findings`` array. Strict by
    default: a finding that omits ``status``/``passed`` is rejected rather than read as a
    silent pass (``--lenient`` opts out). An empty findings set fails closed - an evidence
    set that proves nothing certifies nothing (same rule as ``GateAdjudicator``).
    """
    out = stdout if stdout is not None else sys.stdout
    try:
        if findings_path == "-":
            text = (stdin if stdin is not None else sys.stdin).read()
        else:
            with open(findings_path, encoding="utf-8") as fh:
                text = fh.read()
    except OSError as exc:
        return _fail_closed(f"cannot read findings: {exc}", as_json, out)
    try:
        data = json.loads(text)
    except ValueError as exc:
        return _fail_closed(f"findings are not valid JSON: {exc}", as_json, out)
    if isinstance(data, dict):
        data = data.get("findings")
    if not isinstance(data, list):
        return _fail_closed(
            "findings must be a JSON array (or an object with a 'findings' array)", as_json, out
        )
    if not data:
        return _fail_closed("no findings - an empty evidence set certifies nothing", as_json, out)
    try:
        verdict = compute_verdict(data, strict=not lenient)
    except (TypeError, ValueError) as exc:
        return _fail_closed(f"evidence rejected: {exc}", as_json, out)
    return _emit_verdict(verdict, as_json, out)


def _parse_expect_head(text: str) -> Tuple[int, str]:
    """Parse an external head anchor formatted ``COUNT:HASH``."""
    count_text, sep, head = text.partition(":")
    if not sep or not count_text.isdigit() or not head:
        raise ValueError(f"--expect-head must be COUNT:HASH, got {text!r}")
    return int(count_text), head


def audit_verify_command(
    log_path: str,
    *,
    expect_head: Optional[str] = None,
    as_json: bool = False,
    stdout: Optional[TextIO] = None,
) -> int:
    """Verify a hash-chained audit log. Exit: 0 intact, 1 broken, 2 operational error.

    A missing log fails closed (a missing log is not an intact log), and a nonblank
    line that does not parse as JSON counts as a break: ``recusal.audit.load`` skips
    such a line so a half-written tail cannot brick a *reader*, but a *verifier* that
    ignored it could bless a log whose most recent entries are unreadable.
    """
    out = stdout if stdout is not None else sys.stdout
    expected: Optional[Tuple[int, str]] = None
    if expect_head is not None:
        try:
            expected = _parse_expect_head(expect_head)
        except ValueError as exc:
            return _fail_closed(str(exc), as_json, out)
    if not os.path.exists(log_path):
        return _fail_closed(
            f"audit log not found: {log_path} - a missing log is not an intact log",
            as_json,
            out,
        )
    try:
        entries = _load_audit_entries(log_path)
    except (OSError, UnicodeDecodeError) as exc:
        # operational inability to read is exit 2, not a "broken chain" exit 1
        return _fail_closed(f"cannot read audit log: {exc}", as_json, out)

    # ONE strict verifier for the library and the CLI (recusal.audit.verify_file):
    # malformed nonblank lines, non-object records, and shape violations are failures.
    intact, problems = _verify_audit_file(log_path, expected_head=expected)

    head = GENESIS
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("hash"), str):
            head = entry["hash"]
    if as_json:
        payload = {
            "intact": intact,
            "entries": len(entries),
            "head": head,
            "problems": problems,
            "exit_code": 0 if intact else 1,
        }
        out.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    elif intact:
        out.write(f"intact - {len(entries)} entries, head {head}\n")
    else:
        out.write(f"BROKEN - {len(entries)} entries\n")
        for problem in problems:
            out.write(f"  {problem}\n")
    return 0 if intact else 1


def _launcher_platform_findings(gate_hooks: List[dict]) -> List[Finding]:
    """Validate the registered launchers' SHELL strategy for the host, not just their text.

    The failure mode this catches (live-verified): Claude Code runs shell-form hooks
    under Git Bash on Windows and FALLS BACK TO POWERSHELL when Git Bash is absent -
    where the POSIX launcher is a parse error with exit 1, a NON-blocking code, so the
    gate silently disables. Searching the command for "exit 2" cannot see that.
    """
    findings: List[Finding] = []
    posix = [h for h in gate_hooks if "for p in" in str(h.get("command", ""))]
    powershell = [h for h in gate_hooks if "foreach ($p in" in str(h.get("command", ""))]
    ps_without_shell = [h for h in powershell if h.get("shell") != "powershell"]
    if ps_without_shell:
        findings.append(
            Finding.fail(
                "launcher_shell_strategy",
                severity="CRITICAL",
                message='a PowerShell launcher is registered without "shell": "powershell"; '
                "under Git Bash it is a parse error with a NON-blocking exit code, i.e. the "
                "gate fails OPEN there",
            )
        )
    if _WINDOWS and posix and not powershell:
        if shutil.which("bash"):
            findings.append(
                Finding.fail(
                    "launcher_shell_strategy",
                    severity="WARNING",
                    message="only a POSIX launcher is registered; it works under the "
                    "currently installed Git Bash, but if Git Bash is removed Claude Code "
                    "falls back to PowerShell where this launcher fails OPEN - re-run "
                    "`recusal init` (or use --launcher both) for a PowerShell launcher",
                )
            )
        else:
            findings.append(
                Finding.fail(
                    "launcher_shell_strategy",
                    severity="CRITICAL",
                    message="only a POSIX launcher is registered and no bash is on PATH: "
                    "Claude Code will run it under PowerShell, where it is a parse error "
                    "with a NON-blocking exit code - the gate FAILS OPEN on this host. "
                    "Run `recusal init --repair-launcher` to register the PowerShell launcher",
                )
            )
    if not _WINDOWS and powershell and not posix:
        findings.append(
            Finding.fail(
                "launcher_shell_strategy",
                severity="CRITICAL",
                message="only a PowerShell launcher is registered on a non-Windows host; "
                "run `recusal init --repair-launcher` to register the POSIX launcher",
            )
        )
    if not findings and (posix or powershell):
        findings.append(
            Finding.ok(
                "launcher_shell_strategy",
                severity="WARNING",
                message="registered launcher(s) match this host's shell strategy",
            )
        )
    return findings


def doctor_findings(project_dir: str) -> List[Finding]:
    """Inspect a scaffolded gate installation and return the evidence, as Findings.

    The doctor is adjudicated by the same kernel it checks: findings fold through
    ``compute_verdict``, so a missing gate is a CRITICAL failure, not a log line.
    """
    findings: List[Finding] = []
    claude_dir = os.path.join(project_dir, ".claude")
    gate_path = os.path.join(claude_dir, "hooks", GATE_FILENAME)
    settings_path = os.path.join(claude_dir, "settings.json")

    gate_source_text = ""
    if not os.path.exists(gate_path):
        findings.append(
            Finding.fail(
                "gate_script",
                severity="CRITICAL",
                message=f"no gate at {gate_path} - run `python -m recusal init`",
                path=gate_path,
            )
        )
    else:
        try:
            with open(gate_path, encoding="utf-8") as fh:
                gate_source_text = fh.read()
            compile(gate_source_text, gate_path, "exec")
        except (OSError, SyntaxError, ValueError) as exc:
            findings.append(
                Finding.fail(
                    "gate_script",
                    severity="ERROR",
                    message=f"gate exists but does not compile (the launcher will refuse "
                    f"every call until it is fixed): {exc}",
                    path=gate_path,
                )
            )
        else:
            findings.append(
                Finding.ok(
                    "gate_script",
                    severity="CRITICAL",
                    message="gate present and compiles",
                    path=gate_path,
                )
            )
        if gate_source_text and "recusal" not in gate_source_text:
            findings.append(
                Finding.fail(
                    "gate_imports_recusal",
                    severity="WARNING",
                    message="gate script never references recusal; is it still a recusal gate?",
                )
            )

    if not os.path.exists(settings_path):
        findings.append(
            Finding.fail(
                "hook_registered",
                severity="CRITICAL",
                message=f"{settings_path} does not exist - the gate is not wired into "
                "Claude Code (run `python -m recusal init`)",
            )
        )
    else:
        try:
            with open(settings_path, encoding="utf-8") as fh:
                settings = json.load(fh)
        except ValueError:
            findings.append(
                Finding.fail(
                    "hook_registered",
                    severity="CRITICAL",
                    message=f"{settings_path} does not parse, so no hook in it is registered",
                )
            )
        else:
            gate_hooks = _recusal_hooks(settings if isinstance(settings, dict) else {})
            commands = [str(h["command"]) for h in gate_hooks]
            if not commands:
                findings.append(
                    Finding.fail(
                        "hook_registered",
                        severity="CRITICAL",
                        message="no PreToolUse hook in settings.json references the recusal gate",
                    )
                )
            else:
                findings.append(
                    Finding.ok(
                        "hook_registered",
                        severity="CRITICAL",
                        message=f"{len(commands)} PreToolUse hook(s) reference the gate",
                    )
                )
                if any("exit 2" not in command for command in commands):
                    findings.append(
                        Finding.fail(
                            "launcher_fails_closed",
                            severity="WARNING",
                            message="a registered launcher never exits 2 (the blocking code); "
                            "on a broken interpreter it may fail OPEN",
                        )
                    )
                else:
                    findings.append(
                        Finding.ok(
                            "launcher_fails_closed",
                            severity="WARNING",
                            message="launcher coerces every failure to the blocking exit code",
                        )
                    )
                findings.extend(_launcher_platform_findings(gate_hooks))

    findings.append(
        Finding.ok(
            "recusal_version",
            severity="INFO",
            message=f"recusal {__version__} importable",
        )
    )
    return findings


def doctor_command(
    project_dir: str = ".",
    *,
    as_json: bool = False,
    stdout: Optional[TextIO] = None,
) -> int:
    """Health-check a scaffolded gate. Exit: 0 healthy, 1 degraded, 2 not installed."""
    out = stdout if stdout is not None else sys.stdout
    findings = doctor_findings(project_dir)
    verdict = compute_verdict(findings)
    if as_json:
        payload = _verdict_payload(verdict)
        payload["checks"] = [_finding_brief(f) for f in findings]
        out.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        for f in findings:
            mark = "ok  " if f.passed else "FAIL"
            out.write(f"  [{mark}] {f.check}: {f.message}\n")
        out.write(f"{verdict.decision.value} - {verdict.message}\n")
    return EXIT_BY_DECISION[verdict.decision]


# --- MCP tool and server-instruction integrity: pin the discovery surface, refuse drift ---


class Observation(NamedTuple):
    """What a collection pass gathered, named so the parts don't ride on tuple order.

    - ``catalog``: ``{server: [tool declarations]}`` actually observed;
    - ``notes``: human-readable prose about the collection (printed, never adjudicated);
    - ``unfetchable``: servers declared in the config but unreachable by this fetcher (a
      URL/HTTP transport), handed to ``diff_manifest`` which adjudicates a *pinned*
      unfetchable server into a CRITICAL;
    - ``sources``: each observed server's UNEXPANDED launch template
      (``recusal.mcp.normalize_source`` shape; ``transport: "external"`` for dumps),
      what a pin records and a verify compares BEFORE launching anything;
    - ``instructions``: per server, ``{"observed": bool, "text": str|None}`` - the
      initialize-result instructions when the observation carried them, or
      ``observed: false`` for a legacy dump, which never silently upgrades to the
      stronger discovery-content claim.
    """

    catalog: Dict[str, List[dict]]
    notes: List[str]
    unfetchable: List[str]
    sources: Dict[str, Dict[str, Any]]
    instructions: Dict[str, Dict[str, Any]]


def _atomic_write(path: str, text: str) -> None:
    """Write ``text`` to ``path`` atomically: a crash mid-write never leaves a truncated
    manifest (a half-written pin is approved truth corrupted). Same-directory temp so the
    ``os.replace`` is atomic on one filesystem."""
    directory = os.path.dirname(os.path.abspath(path))
    tmp = os.path.join(directory, f".{os.path.basename(path)}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _collect_catalog(
    *,
    from_file: Optional[str] = None,
    server: Optional[str] = None,
    stdio: Optional[List[List[str]]] = None,
    claude_config: Optional[str] = None,
    timeout: float = 30.0,
    minimal_env: bool = True,
    pinned: Optional[Dict[str, Any]] = None,
) -> Observation:
    """Assemble an :class:`Observation` from the CLI's sources.

    Raises ``ValueError`` / ``OSError`` / :class:`McpFetchError` on anything that prevents a
    complete observation, the caller fails closed. A server named by two sources is an
    error, not a silent override.

    **Pre-launch enforcement**: when ``pinned`` (a loaded manifest) is given, every
    stdio server's launch specification is compared against its pin BEFORE the
    configured command is executed - a changed or never-pinned launch spec raises,
    refusing WITHOUT starting the replacement process. A post-execution catalog
    mismatch proves drift only after the substituted command already ran; this is the
    boundary that proves it first.

    ``Observation.unfetchable`` names servers that ARE declared in the config but this
    fetcher cannot reach (a URL/HTTP transport). That is load-bearing for ``verify``: a
    pinned server silently swapped to a URL transport would otherwise read as merely
    "absent" (a WARNING) and pass; ``diff_manifest`` turns a *pinned* unfetchable server
    into a refusal.
    """
    catalog: Dict[str, List[dict]] = {}
    notes: List[str] = []
    unfetchable: List[str] = []
    sources: Dict[str, Dict[str, Any]] = {}
    instructions: Dict[str, Dict[str, Any]] = {}

    def _add(name: str, tools: List[dict]) -> None:
        if name in catalog:
            raise ValueError(f"server {name!r} is named by more than one source")
        catalog[name] = tools

    def _gate_launch(name: str, source: Dict[str, Any]) -> None:
        """Refuse a launch whose specification is not the approved one - BEFORE it runs."""
        if pinned is None:
            return
        entry = pinned.get("servers", {}).get(name, {})
        findings = diff_source(name, entry if isinstance(entry, dict) else {}, source)
        failed = [f for f in findings if not f.passed]
        if failed:
            raise ValueError(failed[0].message)

    if from_file:
        with open(from_file, encoding="utf-8") as fh:
            data = json.load(fh)
        # The MODE is chosen by --server, not by sniffing a "tools" key: with --server the
        # file is a single server's raw tools/list result; without it, a {server: tools}
        # mapping. This removes the ambiguity where a mapping with a server literally named
        # "tools" was misread as a raw result and its siblings silently dropped.
        if server:
            if isinstance(data, dict) and isinstance(data.get("tools"), list):
                tools = data["tools"]  # a raw tools/list result, or the rich shape
                # The rich single-server shape carries instructions too; the KEY decides
                # the claim: absent = not observed (legacy tools/list result, the weaker
                # claim), present-null = observed and the server declares none, string =
                # observed. Dropping a supplied observation would silently record the
                # weaker claim over a stronger one.
                if "instructions" in data:
                    text = data["instructions"]
                    if text is not None and not isinstance(text, str):
                        raise ValueError(f"{from_file}: 'instructions' must be a string or null")
                    instructions[server] = {"observed": True, "text": text}
                else:
                    instructions[server] = {"observed": False, "text": None}
            elif isinstance(data, list):
                tools = data  # a bare array of tool declarations
                instructions[server] = {"observed": False, "text": None}
            else:
                raise ValueError(
                    f"{from_file} with --server must be a tools/list result (an object with "
                    "a 'tools' array, optionally with 'instructions') or a bare array of "
                    "tool declarations"
                )
            _add(server, tools)
            sources[server] = {"transport": "external"}
        elif isinstance(data, dict) and data:
            # Every key is a server name (a server literally named "tools" is fine here,
            # its siblings are never dropped). A bare tools/list result reaches this branch
            # only if --server was omitted; the per-value list check gives a clear error.
            for name, value in data.items():
                if isinstance(value, list):
                    # legacy shape: tools only - instructions were NOT observed, and
                    # the pin records exactly that weaker claim
                    _add(str(name), value)
                    instructions[str(name)] = {"observed": False, "text": None}
                elif isinstance(value, dict) and isinstance(value.get("tools"), list):
                    # rich shape: {"instructions": str|null, "tools": [...]} - the
                    # observation carries the discovery-content surface too
                    text = value.get("instructions")
                    if text is not None and not isinstance(text, str):
                        raise ValueError(
                            f"{from_file}: server {name!r} 'instructions' must be a string or null"
                        )
                    _add(str(name), value["tools"])
                    instructions[str(name)] = {"observed": True, "text": text}
                else:
                    raise ValueError(
                        f"{from_file}: server {name!r} must map to a tool list, or to "
                        "{'instructions': ..., 'tools': [...]} "
                        "(is this a single server's tools/list result? then pass --server NAME)"
                    )
                sources[str(name)] = {"transport": "external"}
        else:
            raise ValueError(
                f"{from_file} must be a mapping of server -> tools, or (with --server) a "
                "single server's tools/list result"
            )

    for name, command_text in stdio or []:
        argv = split_command(command_text)
        if not argv:
            raise ValueError(f"--stdio {name}: empty command")
        source: Dict[str, Any] = {
            "transport": "stdio",
            "command": argv[0],
            "args": argv[1:],
            "cwd": None,
            "env_templates": {},
        }
        _gate_launch(name, source)
        observed = fetch_server_stdio(argv, timeout=timeout, minimal_env=minimal_env)
        _add(name, observed["tools"])
        sources[name] = source
        instructions[name] = {"observed": True, "text": observed["instructions"]}

    if claude_config:
        stdio_servers, remote_servers = servers_from_claude_config(claude_config)
        # gate EVERY configured server - stdio AND remote - before executing ANY: an
        # unpinned or drifted server of any transport must refuse first, and one bad
        # server must not let its siblings launch before the refusal surfaces
        if pinned is not None:
            for name, spec in stdio_servers.items():
                _gate_launch(name, spec["source"])
            for name, remote_source in remote_servers.items():
                _gate_launch(name, remote_source)
        for name, spec in stdio_servers.items():
            observed = fetch_server_stdio(
                spec["command"],
                env=spec["env"],
                cwd=spec["cwd"],
                timeout=timeout,
                minimal_env=minimal_env,
            )
            _add(name, observed["tools"])
            sources[name] = spec["source"]
            instructions[name] = {"observed": True, "text": observed["instructions"]}
        for name, remote_source in remote_servers.items():
            if name in catalog:
                # catalog supplied via --from under the same name: the DUMP provides the
                # declarations, the CONFIG provides the identity - pin/compare the real
                # remote identity, not a generic "external"
                sources[name] = remote_source
            else:
                unfetchable.append(name)
                sources[name] = remote_source
                notes.append(
                    f"server {name!r} in {claude_config} is remote ({remote_source['transport']});"
                    " this fetcher never contacts it - supply its catalog via --from"
                )

    if not catalog and not unfetchable:
        raise ValueError("no catalog source given (--from, --stdio, or --claude-config)")
    return Observation(catalog, notes, unfetchable, sources, instructions)


#: Templates whose ``${VAR:-default}`` default value is recorded in the manifest; a
#: secret placed in a default is stored just as surely as a literal.
_TEMPLATE_DEFAULT_RE = re.compile(r"\$\{\w+:-([^}]+)\}")

#: Argument flags that conventionally carry a credential in the NEXT argument. A
#: deny-list with a deny-list's ceiling: it surfaces the obvious for review; it is not
#: secret detection, and it never substitutes for reading the source before pinning.
_SECRET_ARG_MARKERS = frozenset(
    {"--api-key", "--apikey", "--token", "--auth-token", "--password", "--secret", "--key"}
)


def _source_review_findings(sources: Dict[str, Dict[str, Any]]) -> List[Finding]:
    """WARNING findings for source material that becomes READABLE manifest content.

    Tool declarations are stored as hashes; source templates are stored in readable
    form so drift can be explained and compared - which means a literal credential in
    an env value, a header value, a ``${VAR:-default}`` default, or a secret-bearing
    command argument is written into the manifest. Each gets a machine-readable
    WARNING (visible under ``--json``) with the fix: reference secrets as ``${VAR}``,
    which pins the reference, never the value.
    """
    findings: List[Finding] = []

    def _template(server: str, what: str, template: str, literal_check: str) -> None:
        if "${" not in template:
            findings.append(
                Finding.fail(
                    literal_check,
                    severity="WARNING",
                    message=f"server {server!r} {what} is a literal value; it is now "
                    "recorded in the manifest - use ${VAR} for secrets",
                    server=server,
                )
            )
            return
        if _TEMPLATE_DEFAULT_RE.search(template):
            findings.append(
                Finding.fail(
                    "mcp_template_default",
                    severity="WARNING",
                    message=f"server {server!r} {what} carries a ${{VAR:-default}} default; "
                    "the default value is recorded in the manifest - avoid secret-bearing "
                    "defaults",
                    server=server,
                )
            )

    for server_name, source in sorted(sources.items()):
        for key, template in sorted(source.get("env_templates", {}).items()):
            _template(server_name, f"env {key!r}", template, "mcp_env_literal")
        for key, template in sorted(source.get("header_templates", {}).items()):
            _template(server_name, f"header {key!r}", template, "mcp_header_literal")
        args = source.get("args") or []
        for i, arg in enumerate(args):
            if (
                arg.lower() in _SECRET_ARG_MARKERS
                and i + 1 < len(args)
                and ("${" not in args[i + 1])
            ):
                findings.append(
                    Finding.fail(
                        "mcp_arg_secret",
                        severity="WARNING",
                        message=f"server {server_name!r} passes a literal value after "
                        f"{arg!r}; it is recorded in the manifest - use ${{VAR}} for "
                        "secrets",
                        server=server_name,
                    )
                )
    return findings


def mcp_pin_command(
    out_path: str,
    *,
    from_file: Optional[str] = None,
    server: Optional[str] = None,
    stdio: Optional[List[List[str]]] = None,
    claude_config: Optional[str] = None,
    timeout: float = 30.0,
    minimal_env: bool = True,
    approve_server_launch: bool = False,
    update: bool = False,
    force: bool = False,
    as_json: bool = False,
    stdout: Optional[TextIO] = None,
) -> int:
    """Pin the observed catalog to a deterministic manifest. Exit 0 pinned, 1 review, 2 refused.

    The pin is the deliberate, human step, so it fails toward refusal: an incomplete
    observation refuses (exit 2), a non-clean review screen (tool declarations, server
    instructions, source configuration warnings) refuses to write until
    ``--force`` records that a human reviewed it (exit 1, RETRY semantics), and an
    existing manifest with different content refuses without ``--update`` (exit 2), a
    pin is approved truth, never silently replaced. Re-pinning identical content is a
    no-op (exit 0).

    Observing a stdio server EXECUTES the configured command - there is no way to ask a
    process for its catalog without running it - so the first pin is an explicit trust
    event: ``--approve-server-launch`` is required exactly when the selected sources
    will execute one or more stdio server commands (a remote-only configuration paired
    with external catalog observations executes nothing and needs no approval),
    recording that a human reviewed the commands about to run (exit 2 without it,
    before anything executes). The pinned manifest then records each server's launch
    specification, and ``verify`` compares it BEFORE launching, so this approval is a
    one-time event per launch spec, not a ritual.
    """
    out = stdout if stdout is not None else sys.stdout
    launches_planned = bool(stdio)
    if claude_config and not launches_planned:
        # parsing is safe (no execution); approval is required only when a stdio
        # process would actually run, so a remote-only config plus --from needs none
        try:
            config_stdio, _ = servers_from_claude_config(claude_config)
        except (OSError, ValueError) as exc:
            return _fail_closed(f"could not read the configuration: {exc}", as_json, out)
        launches_planned = bool(config_stdio)
    if launches_planned and not approve_server_launch:
        return _fail_closed(
            "pinning from --stdio/--claude-config EXECUTES the configured server "
            "commands; review command, args, env templates (and for remote servers the "
            "url, headers, headersHelper, and oauth configuration), then pass "
            "--approve-server-launch to record that a human approved launching them "
            "(nothing was executed)",
            as_json,
            out,
        )
    try:
        obs = _collect_catalog(
            from_file=from_file,
            server=server,
            stdio=stdio,
            claude_config=claude_config,
            timeout=timeout,
            minimal_env=minimal_env,
        )
    except (OSError, ValueError, McpFetchError) as exc:
        # notes accrued before the failure (e.g. URL-skipped servers) are lost on this
        # path; fold the most useful one into the refusal so the operator is not misled.
        return _fail_closed(f"could not observe the catalog: {exc}", as_json, out)
    catalog = obs.catalog
    if not as_json:  # notes are prose; under --json they would corrupt the payload
        for note in obs.notes:
            out.write(f"note: {note}\n")
    if obs.unfetchable:
        # a partial pin reads as a full one: a configured remote server whose catalog
        # was not supplied must refuse the pin, not ride along as a prose note
        return _fail_closed(
            f"server(s) {sorted(obs.unfetchable)} are remote and no catalog was supplied "
            "for them; add --from with their tools/list dump(s) so the pin covers EVERY "
            "configured server, or pin them separately and deliberately",
            as_json,
            out,
        )

    observed_instructions = {
        name: rec["text"] for name, rec in obs.instructions.items() if rec["observed"]
    }
    try:
        text = manifest_to_text(
            build_manifest(catalog, sources=obs.sources, instructions=observed_instructions)
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        # RecursionError: a declaration nested beyond what canonical JSON can serialize
        # is a hostile catalog; a crash is not a verdict, so it refuses like the rest.
        return _fail_closed(f"catalog cannot be pinned: {exc}", as_json, out)

    screen = (
        list(screen_tool_declarations(catalog))
        + screen_server_instructions(observed_instructions)
        + _source_review_findings(obs.sources)
    )
    screen_verdict = compute_verdict(screen)
    if not screen_verdict.passed and not force:
        if as_json:
            payload = _verdict_payload(screen_verdict)
            payload["pinned"] = False
            payload["screen"] = [_finding_brief(f) for f in screen]
            out.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        else:
            _emit_verdict(screen_verdict, as_json, out)
            out.write(
                "not pinned: review the flagged tool declarations, server instructions, "
                "and source configuration warnings; re-run with --force to record that a "
                "human reviewed and accepted them\n"
            )
        return EXIT_BY_DECISION[screen_verdict.decision]
    if not screen_verdict.passed and not as_json:
        for f in screen_verdict.failures:
            out.write(f"reviewed (--force): {f.check}: {f.message}\n")

    n_servers = len(catalog)
    n_tools = sum(len(t) for t in catalog.values())
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            existing = fh.read()
        if existing == text:
            if as_json:
                payload = {
                    "pinned": True,
                    "changed": False,
                    "path": out_path,
                    "servers": n_servers,
                    "tools": n_tools,
                    "exit_code": 0,
                }
                out.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            else:
                out.write(f"already pinned, no change: {out_path}\n")
            return 0
        if not update:
            return _fail_closed(
                f"{out_path} exists with different content - a pin is approved truth; "
                "run `recusal mcp verify` to see the drift, then re-pin deliberately "
                "with --update",
                as_json,
                out,
            )
    _atomic_write(out_path, text)
    if as_json:
        payload = {
            "pinned": True,
            "changed": True,
            "path": out_path,
            "servers": n_servers,
            "tools": n_tools,
            "screen": [_finding_brief(f) for f in screen],
            "exit_code": 0,
        }
        out.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        out.write(f"pinned {n_tools} tool(s) across {n_servers} server(s) -> {out_path}\n")
    return 0


def mcp_verify_command(
    manifest_path: str,
    *,
    from_file: Optional[str] = None,
    server: Optional[str] = None,
    stdio: Optional[List[List[str]]] = None,
    claude_config: Optional[str] = None,
    timeout: float = 30.0,
    minimal_env: bool = True,
    removed: Optional[List[str]] = None,
    as_json: bool = False,
    stdout: Optional[TextIO] = None,
) -> int:
    """Verify the observed catalog against the pin. Exit: 0 match, 2 drift/unpinned/error.

    **Launch identity is verified BEFORE execution**: each configured stdio server's
    source specification (for stdio: unexpanded command/args/cwd/env value templates;
    for remote servers: url template, header templates, and headersHelper, plus - for
    http/sse - the represented OAuth policy fields; ws is header-only) is
    compared against the pin first, and a changed or never-pinned specification refuses
    WITHOUT starting the configured command - the substituted process never runs. Only
    approved specifications are then launched to observe their catalogs.

    A missing manifest fails closed (a missing pin is not a clean pin), and an
    observation that cannot be completed fails closed (a failed fetch must not read as
    "no drift"). Drift findings are CRITICAL, so drift exits 2, the blocking code.

    **Whole-server inventory**: a pinned server absent from the ENTIRE observation is a
    CRITICAL refusal (``mcp_server_unobserved``), because a partial observation must
    not verify clean while the manifest keeps authorizing that server's runtime names.
    A deliberate removal is acknowledged explicitly with ``--removed NAME`` (recorded
    as a passing WARNING); re-pin to make the shrunk server set the approved truth.
    ``--removed`` supports transitions where at least one pinned server remains
    observable: acknowledging EVERY pinned server refuses with a precise message
    (an empty observation certifies nothing, and the manifest keeps authorizing all
    pinned names until it is replaced or removed - no pin, no MCP is the
    decommission path).

    A pinned server that is present in the config but *unfetchable* (silently swapped to a
    URL transport this fetcher cannot reach) is a CRITICAL refusal, not a passing WARNING:
    a pinned capability that can no longer be integrity-checked must not verify clean.
    """
    out = stdout if stdout is not None else sys.stdout
    try:
        pinned = load_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        return _fail_closed(
            f"no usable manifest at {manifest_path!r} ({exc}) - run `recusal mcp pin`",
            as_json,
            out,
        )
    try:
        obs = _collect_catalog(
            from_file=from_file,
            server=server,
            stdio=stdio,
            claude_config=claude_config,
            timeout=timeout,
            minimal_env=minimal_env,
            pinned=pinned,  # launch specs are compared BEFORE any process starts
        )
    except (OSError, ValueError, McpFetchError) as exc:
        return _fail_closed(f"could not observe the catalog: {exc}", as_json, out)

    try:
        # the kernel owns all adjudication AND its composition: diff_observation is the
        # one complete v5 verify (sources + instructions + catalog + unverifiable +
        # removal acknowledgements + whole-server inventory), so the CLI cannot forget
        # a surface the manifest pins.
        findings = diff_observation(
            pinned,
            McpObservation(
                catalog=obs.catalog,
                sources=obs.sources,
                instructions=obs.instructions,
                unverifiable=tuple(obs.unfetchable),
                removed=tuple(removed or ()),
            ),
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        # a malformed observation (e.g. a lone-surrogate string that cannot be canonicalized,
        # or a corrupt pinned manifest) fails closed, never an uncaught traceback / exit 1
        return _fail_closed(f"could not adjudicate the catalog: {exc}", as_json, out)

    if not as_json:
        for note in obs.notes:
            out.write(f"note: {note}\n")
        # affirmative evidence is the point; a clean pass says WHAT matched, not just PASS
        for f in findings:
            if f.passed:
                out.write(f"  [ok] {f.check}: {f.message}\n")
    return _emit_verdict(compute_verdict(findings), as_json, out)


def _add_mcp_source_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--from",
        dest="from_file",
        metavar="FILE",
        help="JSON observation, the escape hatch for HTTP servers. Rich shape (pins "
        "instructions too): {server: {'instructions': str|null, 'tools': [...]}}. "
        "Legacy shapes: {server: [tools]} mapping, or a single server's raw tools/list "
        "result with --server NAME - legacy shapes do NOT establish server-instruction "
        "coverage (recorded as observed: false)",
    )
    p.add_argument(
        "--server",
        help="server name when --from is a single server's observation: a raw "
        "tools/list result, or {'instructions': str|null, 'tools': [...]} to establish "
        "instruction coverage",
    )
    p.add_argument(
        "--stdio",
        nargs=2,
        action="append",
        metavar=("NAME", "COMMAND"),
        help="observe a stdio MCP server live by EXECUTING this command: --stdio github "
        "'npx -y @modelcontextprotocol/server-github' (repeatable)",
    )
    p.add_argument(
        "--claude-config",
        metavar="PATH",
        help="observe every stdio server in a Claude Code .mcp.json by EXECUTING its "
        "declared commands - treat the config as executable code and review it first",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="seconds to wait for a stdio server's initialize/tools/list (default 30)",
    )
    p.add_argument(
        "--minimal-env",
        action="store_true",
        help="the DEFAULT since 0.5.0 (kept for compatibility): launch stdio servers "
        "with a minimal environment - PATH and friends plus the config's own env - "
        "instead of the full shell environment",
    )
    p.add_argument(
        "--inherit-env",
        action="store_true",
        help="opt OUT of the minimal environment and hand stdio servers the full parent "
        "environment (matching how Claude Code launches them); a server being pinned is "
        "not yet trusted with your API keys, so this is the explicit, named trade",
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m recusal",
        description="Recusal: deterministic governance for Claude agents.",
    )
    parser.add_argument("--version", action="version", version=f"recusal {__version__}")
    sub = parser.add_subparsers(dest="command")
    p_init = sub.add_parser(
        "init",
        help="scaffold a Claude Code PreToolUse gate (.claude/hooks + settings.json)",
    )
    p_init.add_argument(
        "--dir",
        default=".",
        help="project directory to scaffold into (default: current directory)",
    )
    p_init.add_argument(
        "--posture",
        choices=["deny-list", "allowlist"],
        default="deny-list",
        help="deny-list: refuse known-bad, defer the rest (broad channels). "
        "allowlist: default-deny, nothing runs unless named (high-stakes channels).",
    )
    p_init.add_argument(
        "--writable-root",
        default="./workspace",
        help="allowlist posture only: the directory subtree writes are confined to",
    )
    p_init.add_argument(
        "--launcher",
        choices=["auto", "posix", "powershell", "both"],
        default="auto",
        help="which fail-closed launcher to register (default auto: PowerShell on "
        "Windows, POSIX elsewhere; 'both' suits a settings.json shared across OSes, at "
        "the cost of the gate running twice per call on Windows hosts with Git Bash)",
    )
    p_init.add_argument(
        "--repair-launcher",
        action="store_true",
        help="replace the CANONICAL recusal launcher(s) already in settings.json with "
        "the right one(s) for this host (the migration path for a pre-0.4.2 Windows "
        "install whose POSIX launcher fails OPEN under the PowerShell fallback); custom "
        "hooks and the gate policy file are never touched",
    )

    p_verdict = sub.add_parser(
        "verdict",
        help="adjudicate a findings JSON file to PASS/RETRY/FAIL (exit 0/1/2), the CI primitive",
    )
    p_verdict.add_argument(
        "findings",
        help="path to a JSON array of findings, or an object with a 'findings' array; "
        "'-' reads stdin",
    )
    p_verdict.add_argument(
        "--json", action="store_true", help="emit the verdict as JSON instead of text"
    )
    p_verdict.add_argument(
        "--lenient",
        action="store_true",
        help="accept findings that omit status/passed as passing, instead of failing closed "
        "(the strict default rejects them so a failure cannot pass silently)",
    )

    p_audit = sub.add_parser("audit", help="operate on a hash-chained audit log")
    audit_sub = p_audit.add_subparsers(dest="audit_command")
    p_verify = audit_sub.add_parser(
        "verify",
        help="verify a JSONL audit log's hash chain (exit 0 intact, 1 broken, 2 error)",
    )
    p_verify.add_argument("log", help="path to the JSONL audit log")
    p_verify.add_argument(
        "--expect-head",
        metavar="COUNT:HASH",
        help="external anchor for the head; catches truncation, tail rewrite, forged appends",
    )
    p_verify.add_argument(
        "--json", action="store_true", help="emit the result as JSON instead of text"
    )

    p_mcp = sub.add_parser(
        "mcp",
        help="pin MCP source templates, server instructions, and tool declarations; "
        "refuse represented drift",
    )
    mcp_sub = p_mcp.add_subparsers(dest="mcp_command")
    p_pin = mcp_sub.add_parser(
        "pin",
        help="pin the observed catalog to a deterministic manifest (exit 0 pinned/"
        "no-op, 1 when tool declarations or server instructions require review, "
        "2 refused)",
    )
    _add_mcp_source_args(p_pin)
    p_pin.add_argument(
        "--out",
        default="mcp-manifest.json",
        help="manifest path to write (default: mcp-manifest.json)",
    )
    p_pin.add_argument(
        "--approve-server-launch",
        action="store_true",
        help="required with --stdio/--claude-config: records that a human reviewed the "
        "configured commands and approves EXECUTING them to observe their catalogs "
        "(observation runs the command; there is no other way to ask a process for its "
        "tools/list). verify then compares each launch spec BEFORE launching.",
    )
    p_pin.add_argument(
        "--update",
        action="store_true",
        help="allow replacing an existing manifest whose content differs (a pin is "
        "approved truth; replacing it is a deliberate step)",
    )
    p_pin.add_argument(
        "--force",
        action="store_true",
        help="proceed after deliberate review: pin even when the screen flags tool "
        "declarations, server instructions, or source configuration warnings, "
        "recording that a human reviewed and accepted them",
    )
    p_pin.add_argument(
        "--json", action="store_true", help="emit the result as JSON instead of text"
    )
    p_mcp_verify = mcp_sub.add_parser(
        "verify",
        help="verify the observed sources, server instructions, and tool declarations "
        "against the pinned manifest (exit 0 match, 2 drift/unpinned/unobserved/error)",
    )
    _add_mcp_source_args(p_mcp_verify)
    p_mcp_verify.add_argument(
        "--manifest",
        default="mcp-manifest.json",
        help="pinned manifest to verify against (default: mcp-manifest.json)",
    )
    p_mcp_verify.add_argument(
        "--removed",
        action="append",
        default=None,
        metavar="NAME",
        help="acknowledge that this pinned server was deliberately removed (repeatable); "
        "without it, a pinned server absent from the whole observation refuses - re-pin "
        "after removal to make the shrunk server set the approved truth. Supports "
        "transitions where at least one pinned server remains observable; to "
        "decommission ALL MCP capability, remove or replace the manifest itself "
        "(no pin, no MCP)",
    )
    p_mcp_verify.add_argument(
        "--json", action="store_true", help="emit the result as JSON instead of text"
    )

    p_doctor = sub.add_parser(
        "doctor",
        help="health-check a scaffolded gate (exit 0 healthy, 1 degraded, 2 not installed)",
    )
    p_doctor.add_argument(
        "--dir",
        default=".",
        help="project directory to inspect (default: current directory)",
    )
    p_doctor.add_argument(
        "--json", action="store_true", help="emit the result as JSON instead of text"
    )

    args = parser.parse_args(argv)

    if args.command == "init":
        if args.repair_launcher:
            return repair_launcher(args.dir, launcher=args.launcher)
        return init(
            args.dir,
            posture=args.posture,
            writable_root=args.writable_root,
            launcher=args.launcher,
        )
    if args.command == "verdict":
        return verdict_command(args.findings, as_json=args.json, lenient=args.lenient)
    if args.command == "audit":
        if args.audit_command == "verify":
            return audit_verify_command(args.log, expect_head=args.expect_head, as_json=args.json)
        p_audit.print_help()
        return 2
    if args.command == "mcp":
        if args.mcp_command == "pin":
            if args.minimal_env and args.inherit_env:
                parser.error("--minimal-env and --inherit-env are mutually exclusive")
            return mcp_pin_command(
                args.out,
                from_file=args.from_file,
                server=args.server,
                stdio=args.stdio,
                claude_config=args.claude_config,
                timeout=args.timeout,
                minimal_env=not args.inherit_env,
                approve_server_launch=args.approve_server_launch,
                update=args.update,
                force=args.force,
                as_json=args.json,
            )
        if args.mcp_command == "verify":
            if args.minimal_env and args.inherit_env:
                parser.error("--minimal-env and --inherit-env are mutually exclusive")
            return mcp_verify_command(
                args.manifest,
                from_file=args.from_file,
                server=args.server,
                stdio=args.stdio,
                claude_config=args.claude_config,
                timeout=args.timeout,
                minimal_env=not args.inherit_env,
                removed=args.removed,
                as_json=args.json,
            )
        p_mcp.print_help()
        return 2
    if args.command == "doctor":
        return doctor_command(args.dir, as_json=args.json)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
