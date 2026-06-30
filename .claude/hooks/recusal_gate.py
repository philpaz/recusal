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
_PROCESS_SUB_TO_SHELL = re.compile(r"\b(sh|bash)\b\s*<\(\s*(curl|wget)\b")
# Piping ANY output into a bare shell interpreter (defeats `... | base64 -d | sh`).
_PIPE_INTO_SHELL = re.compile(r"\|\s*(sh|bash|zsh|dash)\b")
# Destructive commands beyond rm: POSIX (shred, find -delete, truncate -s 0) and
# Windows/PowerShell (rd/rmdir /s, del /s|/q, Remove-Item -Recurse).
_EXTRA_DESTRUCTIVE = re.compile(
    r"\bshred\b"
    r"|\bfind\b[^|&;]*\s-delete\b"
    r"|\btruncate\b[^|&;]*-s\s*0\b"
    r"|\b(rd|rmdir)\b[^|&;]*\s/s\b"
    r"|\bdel\b[^|&;]*\s/[sq]\b"
    r"|\bremove-item\b[^|&;]*-recurse\b"
)
# \S{0,256} (bounded), not \S*, so a long run of '>' can't make this O(n^2) (ReDoS guard).
_REDIRECT_TO_SECRET = re.compile(
    r">>?\s*\S{0,256}(\.env(?:\.[^\s'\"/\\]{1,64})?|\.pem|\.key|\.p12|id_rsa|id_ed25519)"
)
_MAX_CMD_LEN = 4096  # commands longer than this are refused, not adjudicated (DoS guard)
_WRITE_LIKE = re.compile(
    r"\b(tee|sed\s+-i|python\d*\s+-c|perl\s+-e|ruby\s+-e|node\s+-e|cp|mv|copy|xcopy|robocopy|install|rsync|truncate|set-content|add-content|out-file)\b|>>?"
)
_SECRET_PATH_IN_CMD = re.compile(
    r"(\.env(?:\.[^\s'\"/\\]+)?|\.pem\b|\.key\b|\.p12\b|id_rsa\b|id_ed25519\b)"
)

_SECRET_BASENAMES = {".env", "id_rsa", "id_ed25519"}
_SECRET_SUFFIXES = (".pem", ".key", ".p12")
# The gate's own kill-switch: settings that can disable hooks, and the hook scripts.
_SELF_PROTECT = (".claude/settings.json", ".claude/settings.local.json", ".claude/hooks/")


def _norm(cmd: str) -> str:
    return re.sub(r"\s+", " ", cmd).strip().lower()


def _deobfuscate(cmd: str) -> str:
    # Catch simple token-splitting obfuscations: r''m, g""it, cu\rl, rm${IFS}-rf, etc.
    s = cmd.replace("'", "").replace('"', "").replace("`", "").replace("\\", "")
    return re.sub(r"\$\{?ifs\}?", " ", s)  # $IFS / ${IFS} word-splitting -> space


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
        if len(raw) > _MAX_CMD_LEN:
            findings.append(
                Finding.fail(
                    "command_too_long",
                    severity="CRITICAL",
                    message=f"refusing a {len(raw)}-char command the gate cannot adjudicate safely",
                    command=raw[:120],
                )
            )
            return findings
        cmd = _norm(raw)
        cmd_deobf = _deobfuscate(cmd)
        cmd_paths = re.sub(r"/+", "/", cmd.replace("\\", "/"))
        markers = [m for m in _DESTRUCTIVE if m in cmd or m in cmd_deobf]
        if _rm_recursive_force(cmd) or _rm_recursive_force(cmd_deobf):
            markers.append("rm -rf")
        if _CHMOD_WORLD.search(cmd) or _CHMOD_WORLD.search(cmd_deobf):
            markers.append("chmod -R 777")
        if _GIT_FORCE_REFSPEC.search(cmd) or _GIT_FORCE_REFSPEC.search(cmd_deobf):
            markers.append("git push +force")
        if _EXTRA_DESTRUCTIVE.search(cmd) or _EXTRA_DESTRUCTIVE.search(cmd_deobf):
            markers.append("destructive")
        if markers:
            findings.append(
                Finding.fail(
                    "destructive_command",
                    severity="CRITICAL",
                    message=f"refusing destructive command ({', '.join(sorted(set(markers)))})",
                    command=raw,
                )
            )
        if (
            _PIPE_TO_SHELL.search(cmd)
            or _PIPE_TO_SHELL.search(cmd_deobf)
            or _PROCESS_SUB_TO_SHELL.search(cmd)
            or _PROCESS_SUB_TO_SHELL.search(cmd_deobf)
            or _PIPE_INTO_SHELL.search(cmd)
            or _PIPE_INTO_SHELL.search(cmd_deobf)
        ):
            findings.append(
                Finding.fail(
                    "pipe_to_shell",
                    severity="CRITICAL",
                    message="refusing to pipe output straight into a shell interpreter",
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
        if _WRITE_LIKE.search(cmd) and _SECRET_PATH_IN_CMD.search(cmd):
            findings.append(
                Finding.fail(
                    "secret_write_via_bash",
                    severity="CRITICAL",
                    message="refusing a Bash command that appears to write a secret file",
                    command=raw,
                )
            )
        if _WRITE_LIKE.search(cmd) and any(seg in cmd_paths for seg in _SELF_PROTECT):
            findings.append(
                Finding.fail(
                    "self_protection",
                    severity="CRITICAL",
                    message="refusing a Bash command that appears to edit the gate config/hook",
                    command=raw,
                )
            )

    if tool_name in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        path = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
        norm_path = path.replace("\\", "/").lower()
        base = os.path.basename(path).lower()
        low_path = path.lower()
        if (
            base in _SECRET_BASENAMES
            or base.startswith(".env.")
            or low_path.endswith(_SECRET_SUFFIXES)
        ):
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
