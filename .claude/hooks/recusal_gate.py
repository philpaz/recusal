#!/usr/bin/env python3
"""
Recusal governs its own repository.

This is a real Claude Code ``PreToolUse`` hook, registered in
``.claude/settings.json``. When a Claude Code session works on *this* repo, every
tool call is adjudicated here first, destructive shell commands and writes to
secret/protected files are refused before they run, even under bypassPermissions.

The project eats its own dog food: the governance library is the thing governing
its own development. See ``docs/PROVEN.md``.
"""

import os
import sys

# Make `recusal` importable from the repo without an install.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

from recusal import Finding  # noqa: E402
from recusal.claude_code import run_pretooluse_hook  # noqa: E402

# Substrings that mark a destructive, hard-to-reverse shell command.
_DESTRUCTIVE = (
    "rm -rf",
    "git push --force",
    "git push -f",
    "reset --hard",
    "chmod -R 777",
    ":(){",  # fork bomb
    "mkfs",
    "dd if=",
)
_SECRET_BASENAMES = {".env", "id_rsa", "id_ed25519", "LICENSE"}
_SECRET_SUFFIXES = (".pem", ".key", ".p12")


def policy(tool_name: str, tool_input: dict) -> list:
    """Refuse destructive shell and writes to protected/secret files."""
    findings: list = []

    if tool_name == "Bash":
        cmd = str(tool_input.get("command", ""))
        for marker in _DESTRUCTIVE:
            if marker in cmd:
                findings.append(
                    Finding.fail(
                        "destructive_command",
                        severity="CRITICAL",
                        message=f"refusing destructive command containing {marker!r}",
                        command=cmd,
                    )
                )
        if ("curl" in cmd or "wget" in cmd) and ("| sh" in cmd or "| bash" in cmd):
            findings.append(
                Finding.fail(
                    "pipe_to_shell",
                    severity="CRITICAL",
                    message="refusing to pipe a network download straight into a shell",
                    command=cmd,
                )
            )

    if tool_name in ("Write", "Edit", "MultiEdit"):
        path = str(tool_input.get("file_path", ""))
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

    return findings


if __name__ == "__main__":
    run_pretooluse_hook(policy)
