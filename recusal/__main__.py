"""The ``recusal`` command line: scaffold the gate, adjudicate in CI.

Subcommands:

- ``init``: scaffold a fail-closed Claude Code PreToolUse gate (detailed below);
- ``verdict``: adjudicate a findings JSON file into PASS / RETRY / FAIL with blocking
  exit codes (0 / 1 / 2), the CI primitive — any tool can emit findings, recusal
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
import sys
from typing import Any, Dict, List, NamedTuple, Optional, TextIO, Tuple

from . import __version__
from .audit import GENESIS
from .audit import load as _load_audit_entries
from .audit import verify as _verify_audit_chain
from .evidence import Decision, Finding, Verdict, compute_verdict
from .mcp import (
    build_manifest,
    diff_manifest,
    load_manifest,
    manifest_to_text,
    screen_tool_declarations,
)
from .mcp_fetch import (
    McpFetchError,
    fetch_tools_stdio,
    servers_from_claude_config,
    split_command,
)

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


# --- MCP discovery governance: pin the tool catalog, refuse drift -------------------------


class Observation(NamedTuple):
    """What a collection pass gathered, named so the parts don't ride on tuple order.

    - ``catalog``: ``{server: [tool declarations]}`` actually observed;
    - ``notes``: human-readable prose about the collection (printed, never adjudicated);
    - ``unfetchable``: servers declared in the config but unreachable by this fetcher (a
      URL/HTTP transport), handed to ``diff_manifest`` which adjudicates a *pinned*
      unfetchable server into a CRITICAL.
    """

    catalog: Dict[str, List[dict]]
    notes: List[str]
    unfetchable: List[str]


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
) -> Observation:
    """Assemble an :class:`Observation` from the CLI's sources.

    Raises ``ValueError`` / ``OSError`` / :class:`McpFetchError` on anything that prevents a
    complete observation, the caller fails closed. A server named by two sources is an
    error, not a silent override.

    ``Observation.unfetchable`` names servers that ARE declared in the config but this
    fetcher cannot reach (a URL/HTTP transport). That is load-bearing for ``verify``: a
    pinned server silently swapped to a URL transport would otherwise read as merely
    "absent" (a WARNING) and pass; ``diff_manifest`` turns a *pinned* unfetchable server
    into a refusal.
    """
    catalog: Dict[str, List[dict]] = {}
    notes: List[str] = []
    unfetchable: List[str] = []

    def _add(name: str, tools: List[dict]) -> None:
        if name in catalog:
            raise ValueError(f"server {name!r} is named by more than one source")
        catalog[name] = tools

    if from_file:
        with open(from_file, encoding="utf-8") as fh:
            data = json.load(fh)
        # The MODE is chosen by --server, not by sniffing a "tools" key: with --server the
        # file is a single server's raw tools/list result; without it, a {server: tools}
        # mapping. This removes the ambiguity where a mapping with a server literally named
        # "tools" was misread as a raw result and its siblings silently dropped.
        if server:
            if isinstance(data, dict) and isinstance(data.get("tools"), list):
                tools = data["tools"]  # a raw tools/list result object
            elif isinstance(data, list):
                tools = data  # a bare array of tool declarations
            else:
                raise ValueError(
                    f"{from_file} with --server must be a tools/list result (an object with "
                    "a 'tools' array) or a bare array of tool declarations"
                )
            _add(server, tools)
        elif isinstance(data, dict) and data:
            # Every key is a server name (a server literally named "tools" is fine here,
            # its siblings are never dropped). A bare tools/list result reaches this branch
            # only if --server was omitted; the per-value list check gives a clear error.
            for name, tools in data.items():
                if not isinstance(tools, list):
                    raise ValueError(
                        f"{from_file}: server {name!r} must map to a tool list "
                        "(is this a single server's tools/list result? then pass --server NAME)"
                    )
                _add(str(name), tools)
        else:
            raise ValueError(
                f"{from_file} must be a mapping of server -> tools, or (with --server) a "
                "single server's tools/list result"
            )

    for name, command_text in stdio or []:
        _add(name, fetch_tools_stdio(split_command(command_text), timeout=timeout))

    if claude_config:
        servers, skipped = servers_from_claude_config(claude_config)
        for name in skipped:
            unfetchable.append(name)
            notes.append(
                f"server {name!r} in {claude_config} is not a stdio server; this fetcher "
                "cannot reach it - pin it from a JSON dump via --from"
            )
        for name, spec in servers.items():
            _add(name, fetch_tools_stdio(spec["command"], env=spec["env"], timeout=timeout))

    if not catalog and not unfetchable:
        raise ValueError("no catalog source given (--from, --stdio, or --claude-config)")
    return Observation(catalog, notes, unfetchable)


def mcp_pin_command(
    out_path: str,
    *,
    from_file: Optional[str] = None,
    server: Optional[str] = None,
    stdio: Optional[List[List[str]]] = None,
    claude_config: Optional[str] = None,
    timeout: float = 30.0,
    update: bool = False,
    force: bool = False,
    as_json: bool = False,
    stdout: Optional[TextIO] = None,
) -> int:
    """Pin the observed catalog to a deterministic manifest. Exit 0 pinned, 1 review, 2 refused.

    The pin is the deliberate, human step, so it fails toward refusal three ways: an
    incomplete observation refuses (exit 2), a non-clean description screen refuses to
    write until ``--force`` records that a human reviewed it (exit 1, RETRY semantics),
    and an existing manifest with different content refuses without ``--update`` (exit 2),
    a pin is approved truth, never silently replaced. Re-pinning identical content is a
    no-op (exit 0).
    """
    out = stdout if stdout is not None else sys.stdout
    try:
        obs = _collect_catalog(
            from_file=from_file,
            server=server,
            stdio=stdio,
            claude_config=claude_config,
            timeout=timeout,
        )
    except (OSError, ValueError, McpFetchError) as exc:
        # notes accrued before the failure (e.g. URL-skipped servers) are lost on this
        # path; fold the most useful one into the refusal so the operator is not misled.
        return _fail_closed(f"could not observe the catalog: {exc}", as_json, out)
    catalog = obs.catalog
    if not as_json:  # notes are prose; under --json they would corrupt the payload
        for note in obs.notes:
            out.write(f"note: {note}\n")

    try:
        text = manifest_to_text(build_manifest(catalog))
    except (ValueError, UnicodeError) as exc:
        return _fail_closed(f"catalog cannot be pinned: {exc}", as_json, out)

    screen = screen_tool_declarations(catalog)
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
                "not pinned: review the flagged descriptions; re-run with --force to record "
                "that a human reviewed and accepted them\n"
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
    as_json: bool = False,
    stdout: Optional[TextIO] = None,
) -> int:
    """Verify the observed catalog against the pin. Exit: 0 match, 2 drift/unpinned/error.

    A missing manifest fails closed (a missing pin is not a clean pin), and an
    observation that cannot be completed fails closed (a failed fetch must not read as
    "no drift"). Drift findings are CRITICAL, so drift exits 2, the blocking code.

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
        )
    except (OSError, ValueError, McpFetchError) as exc:
        return _fail_closed(f"could not observe the catalog: {exc}", as_json, out)

    try:
        # the kernel owns all adjudication: it turns a pinned-but-unfetchable server (F1)
        # into a CRITICAL, the CLI only collects the `unverifiable` set and prints.
        findings = diff_manifest(pinned, obs.catalog, unverifiable=obs.unfetchable)
    except (ValueError, UnicodeError) as exc:
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
        help="JSON catalog: a mapping of server name -> tool list, or a raw tools/list "
        "result (then pass --server NAME); the escape hatch for HTTP servers",
    )
    p.add_argument(
        "--server",
        help="server name for a --from file that is a single raw tools/list result",
    )
    p.add_argument(
        "--stdio",
        nargs=2,
        action="append",
        metavar=("NAME", "COMMAND"),
        help="observe a stdio MCP server live: --stdio github 'npx -y @modelcontextprotocol/"
        "server-github' (repeatable)",
    )
    p.add_argument(
        "--claude-config",
        metavar="PATH",
        help="observe every stdio server in a Claude Code .mcp.json",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="seconds to wait for a stdio server's initialize/tools/list (default 30)",
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
        help="pin an MCP server tool catalog and refuse drift (discovery governance)",
    )
    mcp_sub = p_mcp.add_subparsers(dest="mcp_command")
    p_pin = mcp_sub.add_parser(
        "pin",
        help="pin the observed catalog to a deterministic manifest "
        "(exit 0 pinned/no-op, 1 descriptions need review, 2 refused)",
    )
    _add_mcp_source_args(p_pin)
    p_pin.add_argument(
        "--out",
        default="mcp-manifest.json",
        help="manifest path to write (default: mcp-manifest.json)",
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
        help="pin even when the description screen flags injection phrasing, recording "
        "that a human reviewed and accepted it",
    )
    p_pin.add_argument(
        "--json", action="store_true", help="emit the result as JSON instead of text"
    )
    p_mcp_verify = mcp_sub.add_parser(
        "verify",
        help="verify the observed catalog against the pinned manifest "
        "(exit 0 match, 2 drift/unpinned/error)",
    )
    _add_mcp_source_args(p_mcp_verify)
    p_mcp_verify.add_argument(
        "--manifest",
        default="mcp-manifest.json",
        help="pinned manifest to verify against (default: mcp-manifest.json)",
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
        return init(args.dir, posture=args.posture, writable_root=args.writable_root)
    if args.command == "verdict":
        return verdict_command(args.findings, as_json=args.json, lenient=args.lenient)
    if args.command == "audit":
        if args.audit_command == "verify":
            return audit_verify_command(args.log, expect_head=args.expect_head, as_json=args.json)
        p_audit.print_help()
        return 2
    if args.command == "mcp":
        if args.mcp_command == "pin":
            return mcp_pin_command(
                args.out,
                from_file=args.from_file,
                server=args.server,
                stdio=args.stdio,
                claude_config=args.claude_config,
                timeout=args.timeout,
                update=args.update,
                force=args.force,
                as_json=args.json,
            )
        if args.mcp_command == "verify":
            return mcp_verify_command(
                args.manifest,
                from_file=args.from_file,
                server=args.server,
                stdio=args.stdio,
                claude_config=args.claude_config,
                timeout=args.timeout,
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
