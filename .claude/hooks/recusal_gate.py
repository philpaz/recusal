#!/usr/bin/env python3
"""
Recusal governs its own repository.

This is a real Claude Code ``PreToolUse`` hook, registered in
``.claude/settings.json``. When a Claude Code session works on *this* repo, every
tool call is adjudicated here first: destructive shell commands, writes to
secret/protected files, and edits to the gate's own configuration are refused
before they run, even under bypassPermissions.

A substring/regex deny-list is a *baseline*, not a guarantee, a determined command
can be obfuscated past any literal matcher, and an allowlist posture is stronger. What
this proves is the seam, an independent gate that refuses before the tool runs and
that guards its own kill-switch, not that this exact list is exhaustive.
"""

import os
import re
import sys

# Make `recusal` importable from the repo without an install.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

from recusal import Finding  # noqa: E402
from recusal.claude_code import run_pretooluse_hook  # noqa: E402

# Markers matched on a whitespace-normalized, lowercased command, so "rm   -rf" == "rm -rf".
_DESTRUCTIVE = (
    "git push --force",
    "git push -f",
    "reset --hard",
    ":(){",  # fork bomb
    "mkfs",
    "dd if=",
    "dd of=",
    "> /dev/sd",
)
_CHMOD_WORLD = re.compile(r"\bchmod\b.*-\w*r.*\b0?777\b")  # recursive chmod to 777
_GIT_FORCE_REFSPEC = re.compile(r"\bgit\s+push\b.*\s\+\S")  # force-push via +refspec
_PIPE_TO_SHELL = re.compile(r"(curl|wget)\b.*(\|\s*(sh|bash)\b|<\(\s*(curl|wget))")
_REDIRECT_TO_SECRET = re.compile(r">>?\s*\S*(\.env|\.pem|\.key|id_rsa|id_ed25519)")

_SECRET_BASENAMES = {".env", "id_rsa", "id_ed25519", "LICENSE"}
_SECRET_SUFFIXES = (".pem", ".key", ".p12")
# The gate's own kill-switch: settings that can disable hooks, and the hook scripts.
_SELF_PROTECT = (".claude/settings.json", ".claude/settings.local.json", ".claude/hooks/")


def _norm(cmd: str) -> str:
    return re.sub(r"\s+", " ", cmd).strip().lower()


def _rm_recursive_force(cmd: str) -> bool:
    """True if the command is an `rm` with both recursive and force, any flag order."""
    if not re.search(r"\brm\b", cmd):
        return False
    short = "".join(re.findall(r"(?:^|\s)-([a-z]+)", cmd))  # bundled short flags
    has_r = "r" in short or "--recursive" in cmd
    has_f = "f" in short or "--force" in cmd
    return has_r and has_f


def policy(tool_name: str, tool_input: dict) -> list:
    """Refuse destructive shell, writes to protected files, and self-disabling edits."""
    findings: list = []

    if tool_name == "Bash":
        raw = str(tool_input.get("command", ""))
        cmd = _norm(raw)
        markers = [m for m in _DESTRUCTIVE if m in cmd]
        if _rm_recursive_force(cmd):
            markers.append("rm -rf")
        if _CHMOD_WORLD.search(cmd):
            markers.append("chmod -R 777")
        if _GIT_FORCE_REFSPEC.search(cmd):
            markers.append("git push +force")
        if markers:
            findings.append(
                Finding.fail(
                    "destructive_command",
                    severity="CRITICAL",
                    message=f"refusing destructive command ({', '.join(sorted(set(markers)))})",
                    command=raw,
                )
            )
        if _PIPE_TO_SHELL.search(cmd):
            findings.append(
                Finding.fail(
                    "pipe_to_shell",
                    severity="CRITICAL",
                    message="refusing to pipe a network download straight into a shell",
                    command=raw,
                )
            )
        if _REDIRECT_TO_SECRET.search(cmd):
            findings.append(
                Finding.fail(
                    "secret_redirect",
                    severity="CRITICAL",
                    message="refusing a shell redirect that writes to a secret file",
                    command=raw,
                )
            )

    if tool_name in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        path = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
        norm_path = path.replace("\\", "/").lower()
        base = os.path.basename(path)
        if base in _SECRET_BASENAMES or base.startswith(".env.") or path.endswith(_SECRET_SUFFIXES):
            findings.append(
                Finding.fail(
                    "protected_file",
                    severity="CRITICAL",
                    message=f"refusing write to a protected/secret file: {path}",
                    path=path,
                )
            )
        if any(seg in norm_path for seg in _SELF_PROTECT):
            findings.append(
                Finding.fail(
                    "self_protection",
                    severity="CRITICAL",
                    message=f"refusing to edit the gate's own config/hook: {path}",
                    path=path,
                )
            )

    return findings


if __name__ == "__main__":
    run_pretooluse_hook(policy)
