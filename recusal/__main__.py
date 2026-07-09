"""The ``recusal`` command line: scaffold the gate, adjudicate in CI.

Subcommands:

- ``init``: scaffold a fail-closed Claude Code PreToolUse gate (detailed below);
- ``verdict``: adjudicate a findings JSON file into PASS / RETRY / FAIL with blocking
  exit codes (0 / 1 / 2), the CI primitive — any tool can emit findings, recusal
  adjudicates them, and a nonzero exit blocks the merge;
- ``audit verify``: check a hash-chained audit log's integrity (``recusal.audit``);
- ``doctor``: health-check a scaffolded gate, so "the gate silently isn't installed"
  is caught by CI instead of discovered during an incident.

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
import sys
from typing import Any, Dict, List, Optional, TextIO, Tuple

from . import __version__
from .audit import GENESIS
from .audit import load as _load_audit_entries
from .audit import verify as _verify_audit_chain
from .evidence import Decision, Finding, Verdict, compute_verdict

#: The fail-closed launcher, verbatim the command this repository registers for itself
#: (see ``.claude/settings.json.example``). Claude Code treats a hook command that cannot
#: launch, or exits with anything other than 2, as a NON-blocking error and lets the tool
#: call proceed; this loop coerces every failure into exit 2 so a broken or absent
#: interpreter refuses the tool call instead of waving it through. (On Windows, Claude
#: Code runs hook commands under Git Bash.)
LAUNCHER_COMMAND = (
    'for p in python3 python py; do "$p" -c \'import sys; sys.exit(0 if sys.version_info'
    ' >= (3, 9) else 1)\' 2>/dev/null && { "$p"'
    ' "$CLAUDE_PROJECT_DIR/.claude/hooks/recusal_gate.py"; rc=$?; [ "$rc" = 0 ] ||'
    " { echo 'recusal gate: hook did not run cleanly; failing closed' >&2; exit 2; };"
    " exit 0; }; done; echo 'recusal gate: no working python>=3.9 interpreter; failing"
    " closed' >&2; exit 2"
)

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

_HOOK_ENTRY = {
    "matcher": ".*",
    "hooks": [{"type": "command", "command": LAUNCHER_COMMAND}],
}


def gate_source(posture: str, writable_root: str = "./workspace") -> str:
    """Return the gate-script source for ``posture`` (``deny-list`` | ``allowlist``)."""
    if posture == "deny-list":
        return _GATE_DENY_LIST
    if posture == "allowlist":
        return _GATE_ALLOWLIST.format(writable_root=writable_root)
    raise ValueError(f"unknown posture: {posture!r}")


def _recusal_hook_commands(settings: dict) -> List[str]:
    """Every PreToolUse command in ``settings`` that references the recusal gate."""
    commands: List[str] = []
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        return commands
    groups = hooks.get("PreToolUse", [])
    if not isinstance(groups, list):
        return commands
    for group in groups:
        if not isinstance(group, dict):
            continue
        for hook in group.get("hooks", []) or []:
            if isinstance(hook, dict) and GATE_FILENAME in str(hook.get("command", "")):
                commands.append(str(hook["command"]))
    return commands


def _has_recusal_hook(settings: dict) -> bool:
    """True if any PreToolUse command already references the recusal gate."""
    return bool(_recusal_hook_commands(settings))


def merge_settings(existing_text: Optional[str]) -> Tuple[Optional[str], str]:
    """Merge the recusal PreToolUse entry into a settings.json body.

    Returns ``(new_text, status)`` where status is one of ``created`` / ``merged`` /
    ``already-installed`` / ``unparseable`` / ``unexpected-shape``. ``new_text`` is
    ``None`` whenever the file must not be written (idempotent no-op or refusal),
    so a caller cannot accidentally clobber a file this function refused to merge.
    """
    if existing_text is None:
        body = {"hooks": {"PreToolUse": [_HOOK_ENTRY]}}
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
    pre.append(_HOOK_ENTRY)
    return json.dumps(settings, indent=2) + "\n", "merged"


def _manual_snippet() -> str:
    entry = {"hooks": {"PreToolUse": [_HOOK_ENTRY]}}
    return json.dumps(entry, indent=2)


def init(
    project_dir: str,
    posture: str = "deny-list",
    writable_root: str = "./workspace",
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
    new_text, status = merge_settings(existing)

    if status in ("unparseable", "unexpected-shape"):
        out.write(
            f"REFUSING to edit {settings_path} ({status}); it was left untouched.\n"
            "Add this PreToolUse entry by hand:\n" + _manual_snippet() + "\n"
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

    entries = _load_audit_entries(log_path)
    intact, problems = _verify_audit_chain(entries, expected_head=expected)

    with open(log_path, encoding="utf-8") as fh:
        nonblank = sum(1 for line in fh if line.strip())
    skipped = nonblank - len(entries)
    if skipped > 0:
        intact = False
        problems.append(
            f"{skipped} nonblank line(s) are not valid JSON - the log's most recent "
            "entries may be unreadable or tampered"
        )

    head = entries[-1].get("hash") if entries else GENESIS
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
            commands = _recusal_hook_commands(settings if isinstance(settings, dict) else {})
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
        return init(args.dir, posture=args.posture, writable_root=args.writable_root)
    if args.command == "verdict":
        return verdict_command(args.findings, as_json=args.json, lenient=args.lenient)
    if args.command == "audit":
        if args.audit_command == "verify":
            return audit_verify_command(args.log, expect_head=args.expect_head, as_json=args.json)
        p_audit.print_help()
        return 2
    if args.command == "doctor":
        return doctor_command(args.dir, as_json=args.json)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
