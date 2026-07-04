#!/usr/bin/env python3
"""
Recusal governs its own repository.

This is a real Claude Code ``PreToolUse`` hook, registered via
``.claude/settings.json`` (copy ``.claude/settings.json.example`` to activate it, a
deliberate step Claude Code asks you to confirm). Once registered, when a Claude Code
session works on *this* repo, every tool call is adjudicated here first: destructive
shell commands, writes to
secret/protected files, and edits *or deletions* of the gate's own configuration
are refused before they run, even under bypassPermissions.

A substring/regex deny-list is a *baseline*, not a guarantee, a determined command
can be obfuscated past any literal matcher, and an allowlist posture is stronger. What
this proves is the seam, an independent gate that refuses before the tool runs and
that guards its own kill-switch, not that this exact list is exhaustive.

Hardening notes (what this hook does that a naive deny-list does not):

- **Uniform de-obfuscation.** Every check, destructive, secret, *and* self-protect,
  runs against both the raw normalized command and a de-obfuscated form (quotes,
  backticks, backslashes, ``$IFS`` word-splitting stripped). Self-protect and secret
  checks additionally run against a path-normalized de-obfuscated form, so
  ``.cla""ude/hooks`` or ``.claude\\hooks`` cannot walk past the kill-switch guard.
- **Delete, not just edit.** The kill-switch is protected against ``rm``/``mv``/``del``
  and friends, not only ``Write``/``Edit`` and shell redirects, deleting the hook
  disables it exactly like editing it.
- **Pipe into any interpreter.** ``| sh`` and ``| python``/``perl``/``ruby``/``node``/
  ``php``/``pwsh`` are all refused, not just POSIX shells.
- **Reverse shells.** ``/dev/tcp`` redirects, ``nc -e`` and ``socat EXEC:`` back-connects refused.
- **Best-effort symlink resolution.** A tool-based write (``Write``/``Edit`` or an MCP
  filesystem tool) whose innocent-looking path resolves through a symlink onto a protected
  control path is refused (``_resolves_into_protected``), closing the classic
  ``notes.txt`` -> ``.claude/settings.json`` TOCTOU. Best-effort: a not-yet-created link can't
  be resolved, and ``Bash`` fragments stay string-matched, so an allowlist is still stronger.

The honest limit is unchanged: a deny-list cannot catch a command whose *name* is
built at runtime (hex/char-codes/``eval`` of decoded data) or code run inside a bare
interpreter (``python script.py``). For high-stakes tools use allowlist mode,
``recusal.claude_code.allowlist_policy`` (see ``docs/COOKBOOK.md`` recipe 11), which
refuses both. That boundary is pinned as a test on each side.
"""

import os
import re
import sys

# Make `recusal` importable from the repo without an install.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

from recusal import Finding  # noqa: E402
from recusal.claude_code import run_pretooluse_hook  # noqa: E402

# Interpreters that execute piped stdin (or a process-substituted download) as code.
_INTERP = r"(?:sh|bash|zsh|dash|ksh|fish|python\d*|perl|ruby|node|php|pwsh|powershell)"

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
# curl/wget piped or process-substituted into ANY interpreter (not just sh/bash).
_PIPE_TO_SHELL = re.compile(r"(curl|wget)\b.*(\|\s*" + _INTERP + r"\b|<\(\s*(curl|wget))")
_PROCESS_SUB_TO_SHELL = re.compile(r"\b" + _INTERP + r"\b\s*<\(\s*(curl|wget)\b")
# Piping ANY output into a bare interpreter (defeats `... | base64 -d | sh|python|...`).
_PIPE_INTO_SHELL = re.compile(r"\|\s*" + _INTERP + r"\b")
# Reverse / bind shells: /dev/tcp back-connect, nc/ncat -e, interactive bash back-connect,
# and socat with an EXEC:/SYSTEM: payload (its shell-spawning form).
_REVERSE_SHELL = re.compile(
    r"/dev/(tcp|udp)/|\b(nc|ncat|netcat)\b[^|&;]{0,256}-\w*e|\b(bash|sh)\b\s+-\w*i\b"
    r"|\bsocat\b[^|&;]{0,256}(exec|system):"
)
# Destructive commands beyond `rm`: POSIX (shred, find -delete / -exec rm, truncate -s 0,
# unlink), git working-tree loss (clean -f, checkout -- , reset --hard), and Windows /
# PowerShell (rd/rmdir /s, del /s|/q, Remove-Item -Recurse).
_EXTRA_DESTRUCTIVE = re.compile(
    r"\bshred\b"
    r"|\bunlink\b"
    r"|\bfind\b[^|&;]{0,256}\s-delete\b"
    r"|\bfind\b[^|&;]{0,256}-exec[^|&;]{0,256}\b(rm|shred|unlink|truncate|dd|mv)\b"
    r"|\btruncate\b[^|&;]{0,256}-s\s*0\b"
    r"|\b(rd|rmdir)\b[^|&;]{0,256}\s/s\b"
    r"|\bdel\b[^|&;]{0,256}\s/[sq]\b"
    r"|\bremove-item\b[^|&;]{0,256}-recurse\b"
    r"|\bgit\s+clean\b[^|&;]{0,256}-[a-z]{0,8}f"  # git clean -f / -fd / -fdx (untracked loss)
    r"|\bgit\s+checkout\b[^|&;]{0,256}\s--(\s|$)"  # git checkout -- (discard working tree)
)
_MAX_CMD_LEN = 4096  # commands longer than this are refused, not adjudicated (DoS guard)
# \S{0,256} (bounded), not \S*, so a long run of '>' can't make this O(n^2) (ReDoS guard).
_REDIRECT_TO_SECRET = re.compile(
    r">>?\s*\S{0,256}(\.env(?:\.[^\s'\"/\\]{1,64})?|\.pem|\.key|\.p12|id_rsa|id_ed25519)"
)
_WRITE_LIKE = re.compile(
    r"\b(tee|sed\s+-i|python\d*\s+-c|perl\s+-e|ruby\s+-e|node\s+-e|cp|mv|copy|xcopy|robocopy|install|rsync|truncate|set-content|add-content|out-file)\b|>>?"
)
_SECRET_PATH_IN_CMD = re.compile(
    r"(\.env(?:\.[^\s'\"/\\]+)?|\.pem\b|\.key\b|\.p12\b|id_rsa\b|id_ed25519\b)"
)
# Verbs that write, move, or delete a path (used to guard the kill-switch against being
# overwritten OR removed). Bare interpreters are excluded: `python .claude/hooks/x.py`
# only *reads* the hook. But their inline-code forms (`python -c`, `perl -e`, ...) can
# open(...,'w') a file, so those are included. `>`/`>>` (redirect-truncate) count.
_SELF_PROTECT_VERB = re.compile(
    r"\b(rm|unlink|shred|truncate|mv|move|cp|copy|xcopy|robocopy|ren|rename|tee|dd|sed"
    r"|install|rsync|ln|mklink|set-content|add-content|out-file|remove-item|del|rd|rmdir"
    r"|chmod|chown|chattr)\b"
    r"|\b(python\d*\s+-c|perl\s+-e|ruby\s+-e|node\s+-e)\b"
    r"|>>?"
)

# `git config core.hooksPath <dir>` redirects git's hooks to an attacker-controlled
# directory -> arbitrary code exec on the next commit/checkout. A gate-disabling vector.
_GIT_HOOK_REDIRECT = re.compile(r"\bgit\s+config\b[^|&;]{0,256}core\.hookspath\b")

_SECRET_BASENAMES = {".env", "id_rsa", "id_ed25519"}
_SECRET_SUFFIXES = (".pem", ".key", ".p12")
# The gate's own kill-switch: settings that can disable hooks (settings.json /
# settings.local.json, matched by the ".claude/settings" prefix) and the hook scripts,
# plus `.git/hooks/` (writing a git hook is another run-code-on-commit vector).
_SELF_PROTECT = (".claude/settings", ".claude/hooks")
_PROTECTED_PATHS = _SELF_PROTECT + (".git/hooks",)

# A non-Bash tool that carries a shell command under one of these keys (an MCP shell, a
# task runner) gets the same command analysis as Bash, so it can't be a second, ungated
# shell. Kept narrow to keys that clearly imply shell execution (low false-positive risk).
# Matched case-insensitively and at ANY nesting depth (see ``_iter_command_values``) so a
# `"Command"` casing or a `{"payload": {"command": ...}}` wrapper can't smuggle a shell past.
_COMMAND_KEYS = frozenset({"command", "cmd", "shell", "script"})
# Built-in tools that only read; they may reference a protected path freely. Any OTHER
# non-Bash tool (Write/Edit, or an MCP filesystem tool) that touches a protected path is
# refused by the generic kill-switch guard. Names are compared lowercased.
_READ_ONLY_TOOLS = frozenset(
    {"read", "glob", "grep", "ls", "notebookread", "webfetch", "websearch", "todowrite", "task"}
)


def _norm(cmd: str) -> str:
    return re.sub(r"\s+", " ", cmd).strip().lower()


def _deobfuscate(cmd: str) -> str:
    # Catch simple token-splitting obfuscations: r''m, g""it, cu\rl, rm${IFS}-rf, etc.
    s = cmd.replace("'", "").replace('"', "").replace("`", "").replace("\\", "")
    return re.sub(r"\$\{?ifs\}?", " ", s)  # $IFS / ${IFS} word-splitting -> space


def _norm_path(s: str) -> str:
    """Path-normalize a command string: backslash -> slash, collapse repeated slashes."""
    return re.sub(r"/+", "/", s.replace("\\", "/"))


def _deobf_path(s: str) -> str:
    """De-obfuscate *and* path-normalize: strip quotes/backticks and $IFS, then treat
    backslashes as separators, so `.cla""ude/hooks` and `.claude\\hooks` both surface as
    `.claude/hooks`. (Distinct from ``_deobfuscate``, which removes backslashes entirely
    for verb matching, that would destroy Windows path separators.)"""
    s = s.replace("'", "").replace('"', "").replace("`", "")
    s = re.sub(r"\$\{?ifs\}?", " ", s)
    return _norm_path(s)


def _resolves_into_protected(path: str) -> bool:
    """Best-effort symlink resolution for the innocent-name -> protected-target TOCTOU case:
    an attacker creates ``notes.txt`` -> ``.claude/settings.json`` and writes the innocent
    *name*, whose string carries no protected segment. Resolve the path and, if a symlink
    lands it on a protected control path, refuse. The resolved path is made relative to the
    working directory before matching, so the repo's own location on disk can't trip a false
    positive. Best-effort by nature: a link that does not exist at hook time cannot be
    resolved, so an allowlist of writable paths stays the real defense (see SECURITY.md)."""
    if not path:
        return False
    try:
        rel = os.path.relpath(os.path.realpath(path), os.getcwd())
    except (OSError, ValueError):
        return False  # unresolvable / different drive -> fall back to the string checks
    norm = _norm_path(rel.lower())
    return any(seg in norm for seg in _PROTECTED_PATHS)


def _search_any(rx: "re.Pattern[str]", variants) -> bool:
    return any(rx.search(v) for v in variants)


def _rm_recursive(cmd: str) -> bool:
    """True if the command is a recursive `rm` (any flag order). Force is not required:
    `rm -r <dir>` destroys a tree just as `rm -rf` does."""
    if not re.search(r"\brm\b", cmd):
        return False
    short = "".join(re.findall(r"(?:^|\s)-([a-z]+)", cmd))  # bundled short flags
    return "r" in short or "--recursive" in cmd


def _iter_strings(obj) -> "list[str]":
    """Every string value nested anywhere in a tool_input (dict/list/scalar)."""
    out: "list[str]" = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_iter_strings(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_iter_strings(v))
    return out


def _iter_command_values(obj) -> "list[str]":
    """Every value that sits under a command-like key (``command``/``cmd``/``shell``/
    ``script``, case-insensitive) anywhere in a tool_input, as a shell string. A list
    value is treated as an argv vector and joined, so ``{"command": ["rm","-rf","/repo"]}``
    is adjudicated exactly like ``"rm -rf /repo"``. This is what stops an MCP shell from
    smuggling a command past the gate via casing (``Command``) or nesting (``args.command``)."""
    out: "list[str]" = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in _COMMAND_KEYS:
                if isinstance(v, str) and v:
                    out.append(v)
                elif isinstance(v, (list, tuple)):
                    joined = " ".join(str(x) for x in v if isinstance(x, (str, int, float)))
                    if joined.strip():
                        out.append(joined)
                else:
                    out.extend(_iter_command_values(v))
            else:
                out.extend(_iter_command_values(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_iter_command_values(v))
    return out


def _analyze_command(raw: str) -> list:
    """Adjudicate a single shell command string. Used for the Bash tool AND for any other
    tool that carries a command under a `_COMMAND_KEYS` field, so an MCP shell can't be a
    second, ungated shell."""
    findings: list = []
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
    # Command variants for verb/marker matching, and path variants for path matching.
    # Three path readings so neither a Windows separator (\ -> /), a quote split, nor a
    # POSIX shell escape (\ dropped) can hide a protected path:
    #   _norm_path(cmd)       -> `.claude\hooks` (Windows) => `.claude/hooks`
    #   _deobf_path(cmd)      -> `.cla""ude/hooks`         => `.claude/hooks`
    #   _norm_path(cmd_deobf) -> `.cl\aude/hooks` (escape) => `.claude/hooks`
    variants = (cmd, cmd_deobf)
    path_variants = (_norm_path(cmd), _deobf_path(cmd), _norm_path(cmd_deobf))

    markers = [m for m in _DESTRUCTIVE if any(m in v for v in variants)]
    if any(_rm_recursive(v) for v in variants):
        markers.append("rm -r")
    if _search_any(_CHMOD_WORLD, variants):
        markers.append("chmod -R 777")
    if _search_any(_GIT_FORCE_REFSPEC, variants):
        markers.append("git push +force")
    if _search_any(_GIT_HOOK_REDIRECT, variants):
        markers.append("git hooksPath redirect")
    if _search_any(_EXTRA_DESTRUCTIVE, variants):
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
        _search_any(_PIPE_TO_SHELL, variants)
        or _search_any(_PROCESS_SUB_TO_SHELL, variants)
        or _search_any(_PIPE_INTO_SHELL, variants)
    ):
        findings.append(
            Finding.fail(
                "pipe_to_shell",
                severity="CRITICAL",
                message="refusing to pipe output straight into a shell/interpreter",
                command=raw,
            )
        )
    if _search_any(_REVERSE_SHELL, variants):
        findings.append(
            Finding.fail(
                "reverse_shell",
                severity="CRITICAL",
                message="refusing a command that looks like a reverse/bind shell",
                command=raw,
            )
        )
    if _search_any(_REDIRECT_TO_SECRET, variants):
        findings.append(
            Finding.fail(
                "secret_redirect",
                severity="CRITICAL",
                message="refusing a shell redirect that writes to a secret file",
                command=raw,
            )
        )
    if _search_any(_WRITE_LIKE, variants) and _search_any(_SECRET_PATH_IN_CMD, variants):
        findings.append(
            Finding.fail(
                "secret_write_via_bash",
                severity="CRITICAL",
                message="refusing a Bash command that appears to write a secret file",
                command=raw,
            )
        )
    if _search_any(_SELF_PROTECT_VERB, variants) and any(
        seg in pv for pv in path_variants for seg in _PROTECTED_PATHS
    ):
        findings.append(
            Finding.fail(
                "self_protection",
                severity="CRITICAL",
                message="refusing a command that edits or removes a protected control path "
                "(gate config/hook or git hooks)",
                command=raw,
            )
        )
    return findings


def policy(tool_name: str, tool_input: dict) -> list:
    """Refuse destructive shell, writes to protected files, and self-disabling edits.

    Coverage is tool-agnostic where it matters: the Bash command analysis also runs on any
    tool that carries a command under a `_COMMAND_KEYS` field (an MCP shell), and a generic
    kill-switch guard refuses any non-read tool (Write/Edit or an MCP filesystem tool) that
    touches a protected control path, so the gate cannot be disabled through a side channel.
    """
    findings: list = []
    tl = tool_name.lower()

    if tool_name == "Bash":
        findings.extend(_analyze_command(str(tool_input.get("command", ""))))
    else:
        # A non-Bash tool that carries a shell command (MCP shell, task runner) is analyzed
        # exactly like Bash, so it cannot become a second, ungated shell. Command-like keys
        # are found case-insensitively and at any nesting depth, and argv arrays are joined.
        for val in _iter_command_values(tool_input):
            findings.extend(_analyze_command(val))

    if tool_name in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        path = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
        base = os.path.basename(path).lower()
        low_path = path.lower()
        if (
            base in _SECRET_BASENAMES
            or base.startswith(".env.")
            or base.endswith(".env")  # .env, prod.env, production.env, ...
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
        elif not any(
            seg in _norm_path(low_path) or seg in _deobf_path(low_path) for seg in _PROTECTED_PATHS
        ) and _resolves_into_protected(path):
            # Innocent name, protected target: a symlink resolves the write onto the
            # gate's own config/hook. Only reached when the *literal* path carries no
            # protected segment (a direct reference is caught by the kill-switch guard
            # below with an accurate message), so the "via a symlink" reason is truthful
            # rather than firing on a plain path that merely lands on a protected target.
            findings.append(
                Finding.fail(
                    "self_protection",
                    severity="CRITICAL",
                    message=f"refusing a write whose path resolves via a symlink onto a "
                    f"protected control path: {path}",
                    path=path,
                )
            )

    # Generic kill-switch guard: any non-Bash tool that is not a known read-only builtin
    # (Write/Edit, or an arbitrary MCP filesystem tool) is refused if ANY of its string
    # inputs references a protected control path, in either path reading. Bash is excluded
    # (its verb-gated analysis correctly allows reads like `cat .claude/settings.json`).
    if tool_name != "Bash" and tl not in _READ_ONLY_TOOLS:
        for s in _iter_strings(tool_input):
            low = s.lower()
            # A path-like string also gets best-effort symlink resolution, so an MCP
            # filesystem tool can't reach a protected target through an innocent-named link,
            # including a *bare* filename with no separator (`notes.txt` -> `.claude/
            # settings.json`), which a separator-only guard misses even though `Write`
            # resolves it. Path-like = short and either carries a separator (dir/file, may
            # contain spaces) OR is a single whitespace-free token (a bare filename). Prose/
            # content blobs carry internal whitespace and no separator, so they are excluded
            # from the stat walk.
            stripped = s.strip()
            path_like = (
                len(s) <= 1024
                and stripped != ""
                and ("/" in stripped or "\\" in stripped or not re.search(r"\s", stripped))
            )
            if any(
                seg in _norm_path(low) or seg in _deobf_path(low) for seg in _PROTECTED_PATHS
            ) or (path_like and _resolves_into_protected(s)):
                findings.append(
                    Finding.fail(
                        "self_protection",
                        severity="CRITICAL",
                        message=f"refusing a `{tool_name}` call that targets a protected "
                        f"control path (gate config/hook or git hooks): {s}",
                        path=s,
                    )
                )
                break

    return findings


if __name__ == "__main__":
    run_pretooluse_hook(policy)
