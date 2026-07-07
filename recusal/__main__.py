"""``python -m recusal init``: scaffold the Claude Code gate in one command.

The README's manual path (hand-editing ``.claude/settings.json`` and writing a gate
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
from typing import List, Optional, Tuple

#: The fail-closed launcher, verbatim the command this repository registers for itself
#: (see ``.claude/settings.json.example``). Claude Code treats a hook command that cannot
#: launch, or exits with anything other than 2, as a NON-blocking error and lets the tool
#: call proceed; this loop coerces every failure into exit 2 so a broken or absent
#: interpreter refuses the tool call instead of waving it through. (On Windows, Claude
#: Code runs hook commands under Git Bash.)
LAUNCHER_COMMAND = (
    "for p in python3 python py; do \"$p\" -c 'import sys; sys.exit(0 if sys.version_info"
    " >= (3, 9) else 1)' 2>/dev/null && { \"$p\""
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


def _has_recusal_hook(settings: dict) -> bool:
    """True if any PreToolUse command already references the recusal gate."""
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        return False
    groups = hooks.get("PreToolUse", [])
    if not isinstance(groups, list):
        return False
    for group in groups:
        if not isinstance(group, dict):
            continue
        for hook in group.get("hooks", []) or []:
            if isinstance(hook, dict) and GATE_FILENAME in str(hook.get("command", "")):
                return True
    return False


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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m recusal",
        description="Recusal: deterministic governance for Claude agents.",
    )
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
    args = parser.parse_args(argv)

    if args.command != "init":
        parser.print_help()
        return 2
    return init(args.dir, posture=args.posture, writable_root=args.writable_root)


if __name__ == "__main__":
    sys.exit(main())
