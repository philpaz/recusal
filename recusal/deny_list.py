"""
Reference deny-list policy: refuse known-bad tool calls, defer the rest.

This is the battle-tested command/path guard that governs Recusal's own repository
(the ``.claude/hooks/recusal_gate.py`` dogfood hook is now a thin shim over this
module). It is importable, versioned, and unit-tested here so a fix ships through
``pip install -U`` instead of a copy-paste, and so the security logic is covered as
a package unit rather than only end-to-end.

It is a **deny-list**: a *baseline* that refuses destructive shell, writes to secret
files, and edits/deletions of a gate's own control paths, then defers everything else.
A deny-list is not a guarantee, a determined command can be obfuscated past any literal
matcher, and code inside a bare interpreter (``python script.py``) is unreadable to it.
For a narrow, high-stakes channel, prefer :func:`recusal.claude_code.allowlist_policy`
(default-deny), which refuses both. The channel decides; this module owns the deny-list
side of that choice.

Usage (matches the dogfood hook)::

    from recusal.claude_code import run_pretooluse_hook
    from recusal.deny_list import deny_list_policy

    run_pretooluse_hook(deny_list_policy())

Point it at *your* gate's control paths::

    run_pretooluse_hook(deny_list_policy(protected_paths=(".mygate/", ".git/hooks")))

Hardening this does that a naive deny-list does not: uniform de-obfuscation (quotes,
backticks, backslashes, ``$IFS`` stripped) across destructive/secret/self-protect
checks; delete-not-just-edit protection of the kill-switch; pipe-into-*any*-interpreter;
reverse/bind shells; ``cd``/variable indirection onto a control dir; package-manager
mutation of the enforcement package itself (``pip uninstall recusal``, install-time
shadowing, across the ``pip`` / ``python -m pip`` / ``uv`` spellings); and best-effort
symlink resolution for the innocent-name -> protected-target TOCTOU. The honest limit is
unchanged: a runtime-constructed command name or code inside a bare interpreter is a
deny-list ceiling, use allowlist mode for those.

Pure logic, standard library only.
"""

import os
import re
from functools import lru_cache
from typing import Any, Callable, FrozenSet, Iterable, List, Sequence, Tuple

from .evidence import Finding

# A policy maps a proposed tool call to evidence findings (matches recusal.claude_code.Policy).
Policy = Callable[[str, dict], List[Any]]

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
# Fork bomb, tolerant of the spacing the literal ":(){" marker misses: a function
# whose body pipes itself into a backgrounded copy (":(){ :|:& };:" and its spaced
# "\: () { : | : & } ; :" form). A *renamed* bomb is a runtime-name deny-list ceiling.
_FORK_BOMB = re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&")
_CHMOD_WORLD = re.compile(
    r"\bchmod\b[^|&;]{0,256}-\w*r[^|&;]{0,256}\b0?777\b"
)  # recursive chmod to 777
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
    r"|\bgit\s+restore(?![\w-])"  # git restore (discard/overwrite tree); not git-restore-mtime
)
_MAX_CMD_LEN = 4096  # commands longer than this are refused, not adjudicated (DoS guard)
# \S{0,256} (bounded), not \S*, so a long run of '>' can't make this O(n^2) (ReDoS guard).
_REDIRECT_TO_SECRET = re.compile(
    r">>?\s*\S{0,256}(\.env(?:\.[^\s'\"/\\]{1,64})?|\.pem|\.key|\.p12|id_rsa|id_ed25519)"
)
_WRITE_LIKE = re.compile(
    r"\b(tee|sed\s+-i|python\d*\s+-c|perl\s+-e|ruby\s+-e|node\s+-e|cp|mv|copy|xcopy|robocopy|install|rsync|truncate|set-content|add-content|clear-content|out-file|new-item)\b|\[?io\.file\]?::\w{0,64}(write|append)|>>?"
)
_SECRET_PATH_IN_CMD = re.compile(
    r"(\.env(?:\.[^\s'\"/\\]+)?|\.pem\b|\.key\b|\.p12\b|id_rsa\b|id_ed25519\b)"
)
# Verbs that write, move, or delete a path (used to guard the kill-switch against being
# overwritten OR removed). Bare interpreters are excluded: `python .claude/hooks/x.py`
# only *reads* the hook. But their inline-code forms (`python -c`, `perl -e`, ...) can
# open(...,'w') a file, so those are included. `>`/`>>` (redirect-truncate) count.
_SELF_PROTECT_VERB = re.compile(
    r"\b(rm|unlink|shred|truncate|mv|move|cp|copy|xcopy|robocopy|ren|rename|tee|dd"
    r"|install|uninstall|rsync|ln|mklink|set-content|add-content|clear-content|out-file|new-item"
    r"|set-itemproperty|remove-item|del|rd|rmdir|chmod|chown|chattr|attrib|takeown|icacls)\b"
    r"|\bsed\s+-\w*i"  # sed -i (in-place write); sed -n / print-to-stdout is a read
    # Inline interpreter code (`... -c`/`-e`/`eval`) can open(...,'w') a protected file.
    # Generalized past the bare `python -c` form: `py` (the Windows launcher), a dotted
    # version (`python3.12 -c`), `node --eval`, and the `deno`/`bun eval` subcommands.
    r"|\b(?:py|python[\d.]*|pypy[\d.]*)\s+-\w*c\b"
    r"|\b(?:perl|ruby)\s+-\w*e\b"
    r"|\bnode\s+(?:-e|--eval)\b"
    r"|\b(?:deno|bun)\s+eval\b"
    r"|\b(?:pwsh|powershell)\s+-[ce]\w*\b"
    r"|\bosascript\s+-e\b"
    # More inline-exec interpreters with a no-redirect file-write builtin (php
    # file_put_contents, lua io.open('w'), R writeLines): same primitive as
    # `python -c`, so a `php -r '...'` writing the kill-switch is covered too.
    r"|\bphp\s+-\w*r\b"
    r"|\b(?:lua|luajit)\s+-\w*e\b"
    r"|\brscript\s+-\w*e\b"
    r"|\b(?:groovy|elixir)\s+-\w*e\b"
    r"|\[?io\.file\]?::\w{0,64}(write|append)"
    r"|>>?"
)

# `git config core.hooksPath <dir>` redirects git's hooks to an attacker-controlled
# directory -> arbitrary code exec on the next commit/checkout. A gate-disabling vector.
_GIT_HOOK_REDIRECT = re.compile(r"\bgit\s+config\b[^|&;]{0,256}core\.hookspath\b")

# Moving, renaming, copying over, or removing a *control directory* itself -- `.claude`,
# `.git`, or the `recusal` enforcement package -- disables the gate just like editing its
# contents, but the bare directory name carries none of the `.claude/hooks`- or `recusal/`
# -style segments matched elsewhere. Catch move/remove verbs aimed at the directory token
# itself. `.gitignore`/`.github`/`recusal_foo` are excluded by the trailing lookahead (which
# also rejects a following `.`, so `recusal.egg-info` is not matched); a longer path like
# `.claude/hooks` is still (harmlessly) re-matched here and by the segment guard. These
# directory-token regexes are tuned to the standard control dirs (.claude/.git/recusal);
# override `protected_paths` for the substring checks, and these still cover the universal
# .claude/.git for any Claude Code adopter.
_CONTROL_DIR_OP = re.compile(
    r"\b(mv|move|ren|rename|cp|copy|xcopy|robocopy|rm|rmdir|rd|del|remove-item|ln|mklink)\b"
    # `/` is excluded from neither look-around, so a leading `./`/`../` or a trailing `/`
    # (`mv ./recusal x`, `mv .claude/ x`) no longer exempts the control-dir token; `.` stays
    # excluded so `foo.recusal`/`recusal.egg-info` don't match, `-` so `recusal-docs` doesn't.
    r"[^|&;]{0,256}(?<![\w.])(?:\.(?:claude|git)|recusal)(?![\w.-])"
)

# `cd .claude && rm settings.json` splits the protected path across the `&&`: the
# self-protect check matches `.claude/settings` as a contiguous substring, but here the
# directory and the filename never touch. Catch the two ways the control dir becomes the
# (implicit) write target without appearing as one token: changing INTO it (`cd`/`pushd`)
# or binding it to a variable (`d=.claude; rm $d/settings.json`). Combined with a write
# verb elsewhere in the command (see the self_protection check), that is a kill-switch
# edit. A read after `cd` (`cd .claude && cat settings.json`) carries no write verb and
# still defers. Fully computed paths (command substitution) remain the deny-list ceiling.
_CD_INTO_CONTROL = re.compile(
    r"\b(?:cd|pushd)\s+[\"']?\.?/?(?:\.(?:claude|git)|recusal)(?![\w.-])"
    r"|(?:^|[\s;&|(])\w+=[\"']?\.?/?(?:\.(?:claude|git)|recusal)(?![\w.-])"
)

# Package-manager mutation of the enforcement package itself. `pip uninstall recusal`
# removes the package every gate hook imports, and an install-time replacement
# (`pip install -e ./fake-recusal`, a URL install, a pinned-version reinstall) swaps
# what the gate IS: neither carries a `recusal/` path segment, so the path-based
# self-protect guard never sees them ("uninstall" also never matched `\binstall\b`,
# there is no word boundary inside the word). Covered spellings: `pip` / `pip3` /
# `pip3.12`, `python -m pip` (plus `py` / `pypy`, with an optional `py -3.12`
# selector), `uv pip`, and `uv add` / `uv remove`; bounded gaps allow flags in
# either position (`pip -q install`, `uninstall -y`). The named ceiling is
# unchanged: an install that provides the package without naming it
# (`pip install -e .`, `-r requirements.txt`) is unreadable to a string matcher;
# a protected, pinned venv stays the real defense (SECURITY.md).
_PIP_SPELLINGS = (
    r"\b(?:pip[\d.]*|uv\s+pip|(?:py|python[\d.]*|pypy[\d.]*)\s+(?:-[\d.]+\s+)?-m\s+pip[\d.]*)"
)


@lru_cache(maxsize=32)
def _package_mutation_rx(packages: Tuple[str, ...]) -> "re.Pattern[str]":
    names = "|".join(re.escape(p) for p in packages)
    return re.compile(
        _PIP_SPELLINGS
        + r"(?:\s+-[^\s|&;]{1,64}){0,6}\s+(?:un)?install\b[^|&;]{0,256}(?:"
        + names
        + r")"
        + r"|\buv\s+(?:add|remove)\b[^|&;]{0,256}(?:"
        + names
        + r")"
    )


# ── Default configuration (the dogfood gate's layout) ─────────────────────────────────
DEFAULT_SECRET_BASENAMES: FrozenSet[str] = frozenset({".env", "id_rsa", "id_ed25519"})
DEFAULT_SECRET_SUFFIXES: Tuple[str, ...] = (".pem", ".key", ".p12")
# The gate's own kill-switch: settings that can disable hooks (settings.json /
# settings.local.json, matched by the ".claude/settings" prefix), the hook scripts, and
# the in-repo ``recusal/`` package a dogfood hook imports and delegates every decision to.
# Editing ``recusal/*.py`` is equivalent to editing the hook; plus ``.git/hooks/`` (writing
# a git hook is another run-code-on-commit vector). Override for your own gate's paths.
DEFAULT_SELF_PROTECT: Tuple[str, ...] = (".claude/settings", ".claude/hooks", "recusal/")
# `.mcp.json` and `mcp-manifest.json` are control-plane artifacts of the same rank as the
# hook itself: the config decides which server processes launch, and the manifest is the
# approved truth `manifest_policy` reloads at call time - an agent that can rewrite
# either can change what "approved" means before its next tool call.
DEFAULT_PROTECTED_PATHS: Tuple[str, ...] = DEFAULT_SELF_PROTECT + (
    ".git/hooks",
    ".mcp.json",
    "mcp-manifest.json",
)
# The distribution the gate imports, protected at the package-manager layer: uninstalling,
# reinstalling, or shadowing it changes the enforcement code without any protected path
# segment appearing in the command. Manage the gate's own package outside the governed
# session (the README's pinned-venv deployment); override for a gate built on another
# package name.
DEFAULT_PROTECTED_PACKAGES: Tuple[str, ...] = ("recusal",)
# A non-Bash tool that carries a shell command under one of these keys (an MCP shell, a
# task runner) gets the same command analysis as Bash, so it can't be a second, ungated
# shell. Kept narrow to keys that clearly imply shell execution (low false-positive risk).
DEFAULT_COMMAND_KEYS: FrozenSet[str] = frozenset({"command", "cmd", "shell", "script"})
# Built-in tools that only read; they may reference a protected path freely. Any OTHER
# non-Bash tool (Write/Edit, or an MCP filesystem tool) that touches a protected path is
# refused by the generic kill-switch guard. Names are compared lowercased.
DEFAULT_READ_ONLY_TOOLS: FrozenSet[str] = frozenset(
    {"read", "glob", "grep", "ls", "notebookread", "webfetch", "websearch", "todowrite", "task"}
)
_WRITE_TOOLS: Tuple[str, ...] = ("Write", "Edit", "MultiEdit", "NotebookEdit")


def _norm(cmd: str) -> str:
    return re.sub(r"\s+", " ", cmd).strip().lower()


def _deobfuscate(cmd: str) -> str:
    # Catch simple token-splitting obfuscations: r''m, g""it, cu\rl, rm${IFS}-rf, etc.
    s = cmd.replace("'", "").replace('"', "").replace("`", "").replace("\\", "")
    return re.sub(r"\$\{?ifs\}?", " ", s)  # $IFS / ${IFS} word-splitting -> space


def _norm_path(s: str) -> str:
    """Path-normalize a command string: backslash -> slash, collapse repeated slashes, and
    drop trailing dots/spaces from each path component. Windows strips trailing dots and
    spaces from a name, so `recusal./x` and `recusal /x` both resolve into the `recusal`
    package; normalizing them keeps a protected segment from being hidden behind a trailing
    dot (biases toward refusal, never away)."""
    s = re.sub(r"/+", "/", s.replace("\\", "/"))
    # rstrip each '/'-delimited component: linear, unlike a lookahead like `[ .]+(?=/)`
    # which backtracks quadratically on a long run of dots/spaces (a ReDoS foot-gun).
    return "/".join(part.rstrip(" .") for part in s.split("/"))


def _deobf_path(s: str) -> str:
    """De-obfuscate *and* path-normalize: strip quotes/backticks and $IFS, then treat
    backslashes as separators, so `.cla""ude/hooks` and `.claude\\hooks` both surface as
    `.claude/hooks`. (Distinct from ``_deobfuscate``, which removes backslashes entirely
    for verb matching, that would destroy Windows path separators.)"""
    s = s.replace("'", "").replace('"', "").replace("`", "")
    s = re.sub(r"\$\{?ifs\}?", " ", s)
    return _norm_path(s)


def _resolves_into_protected(path: str, protected_paths: Sequence[str]) -> bool:
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
    return any(seg in norm for seg in protected_paths)


def _search_any(rx: "re.Pattern[str]", variants: Iterable[str]) -> bool:
    return any(rx.search(v) for v in variants)


def _rm_recursive(cmd: str) -> bool:
    """True if the command is a recursive `rm` (any flag order). Force is not required:
    `rm -r <dir>` destroys a tree just as `rm -rf` does."""
    if not re.search(r"\brm\b", cmd):
        return False
    short = "".join(re.findall(r"(?:^|\s)-([a-z]+)", cmd))  # bundled short flags
    return "r" in short or "--recursive" in cmd


def _iter_strings(obj: Any) -> List[str]:
    """Every string value nested anywhere in a tool_input (dict/list/scalar)."""
    out: List[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_iter_strings(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_iter_strings(v))
    return out


def _iter_command_values(obj: Any, command_keys: FrozenSet[str]) -> List[str]:
    """Every value that sits under a command-like key (``command``/``cmd``/``shell``/
    ``script``, case-insensitive) anywhere in a tool_input, as a shell string. A list
    value is treated as an argv vector and joined, so ``{"command": ["rm","-rf","/repo"]}``
    is adjudicated exactly like ``"rm -rf /repo"``. This is what stops an MCP shell from
    smuggling a command past the gate via casing (``Command``) or nesting (``args.command``)."""
    out: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in command_keys:
                if isinstance(v, str) and v:
                    out.append(v)
                elif isinstance(v, (list, tuple)):
                    joined = " ".join(str(x) for x in v if isinstance(x, (str, int, float)))
                    if joined.strip():
                        out.append(joined)
                else:
                    out.extend(_iter_command_values(v, command_keys))
            else:
                out.extend(_iter_command_values(v, command_keys))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_iter_command_values(v, command_keys))
    return out


def analyze_command(
    raw: str,
    protected_paths: Sequence[str] = DEFAULT_PROTECTED_PATHS,
    protected_packages: Sequence[str] = DEFAULT_PROTECTED_PACKAGES,
) -> List[Finding]:
    """Adjudicate a single shell command string into a list of Findings (empty = no
    objection). Used for the Bash tool AND for any other tool that carries a command under a
    command-like key, so an MCP shell can't be a second, ungated shell. ``protected_paths``
    are the control-path substrings whose edit/removal is refused; ``protected_packages``
    are the distribution names whose package-manager mutation is refused."""
    findings: List[Finding] = []
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
    if _search_any(_FORK_BOMB, variants):
        markers.append("fork bomb")
    if _search_any(_CHMOD_WORLD, variants):
        markers.append("chmod -R 777")
    if _search_any(_GIT_FORCE_REFSPEC, variants):
        markers.append("git push +force")
    if _search_any(_GIT_HOOK_REDIRECT, variants):
        markers.append("git hooksPath redirect")
    if _search_any(_CONTROL_DIR_OP, variants):
        markers.append("control-directory move/remove")
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
    if protected_packages and _search_any(
        _package_mutation_rx(tuple(protected_packages)), variants
    ):
        findings.append(
            Finding.fail(
                "package_self_protection",
                severity="CRITICAL",
                message="refusing a package-manager command that uninstalls, reinstalls, "
                "or shadows the gate's enforcement package "
                f"({', '.join(protected_packages)}); manage it outside the governed session",
                command=raw,
            )
        )
    targets_protected = any(
        seg in pv for pv in path_variants for seg in protected_paths
    ) or _search_any(_CD_INTO_CONTROL, variants)
    if _search_any(_SELF_PROTECT_VERB, variants) and targets_protected:
        findings.append(
            Finding.fail(
                "self_protection",
                severity="CRITICAL",
                message="refusing a command that edits or removes a protected control path "
                "(gate config, hook, the recusal enforcement package, or git hooks)",
                command=raw,
            )
        )
    return findings


def deny_list_policy(
    *,
    protected_paths: Sequence[str] = DEFAULT_PROTECTED_PATHS,
    protected_packages: Sequence[str] = DEFAULT_PROTECTED_PACKAGES,
    secret_basenames: Iterable[str] = DEFAULT_SECRET_BASENAMES,
    secret_suffixes: Sequence[str] = DEFAULT_SECRET_SUFFIXES,
    command_keys: Iterable[str] = DEFAULT_COMMAND_KEYS,
    read_only_tools: Iterable[str] = DEFAULT_READ_ONLY_TOOLS,
) -> Policy:
    """Build the reference deny-list :data:`Policy`: refuse destructive shell, writes to
    secret files, and self-disabling edits of the gate's control paths, then defer the rest.

    Coverage is tool-agnostic where it matters: the Bash command analysis also runs on any
    tool that carries a command under a command-like key (an MCP shell), and a generic
    kill-switch guard refuses any non-read tool (Write/Edit or an MCP filesystem tool) that
    touches a protected control path, so the gate cannot be disabled through a side channel.

    - ``protected_paths``: control-path substrings whose edit/removal is refused (default:
      the dogfood gate's ``.claude/settings``, ``.claude/hooks``, ``recusal/``, ``.git/hooks``,
      plus the MCP control plane: ``.mcp.json`` and ``mcp-manifest.json``, since rewriting
      the server config or the pinned manifest changes what "approved" means).
    - ``protected_packages``: distribution names whose package-manager mutation is refused
      (default: ``recusal``, the package the gate imports; ``pip uninstall recusal`` or an
      install-time shadow changes the enforcement code without touching a protected path).
    - ``secret_basenames`` / ``secret_suffixes``: files a write tool may not create/overwrite.
    - ``command_keys``: keys under which a non-Bash tool's shell command is found and analyzed.
    - ``read_only_tools``: tools exempt from the generic kill-switch guard (they only read).

    Plug the result into :func:`recusal.claude_code.run_pretooluse_hook`. A clean call
    *defers* to the host's normal permission flow; this policy only ever adds refusals.
    """
    protected = tuple(protected_paths)
    protected_pkgs = tuple(protected_packages)
    secret_bases = frozenset(secret_basenames)
    secret_sfx = tuple(secret_suffixes)
    cmd_keys = frozenset(command_keys)
    readonly = frozenset(t.lower() for t in read_only_tools)

    def policy(tool_name: str, tool_input: dict) -> List[Any]:
        findings: List[Any] = []
        tl = tool_name.lower()

        if tool_name == "Bash":
            findings.extend(
                analyze_command(str(tool_input.get("command", "")), protected, protected_pkgs)
            )
        else:
            # A non-Bash tool that carries a shell command (MCP shell, task runner) is analyzed
            # exactly like Bash, so it cannot become a second, ungated shell. Command-like keys
            # are found case-insensitively and at any nesting depth, and argv arrays are joined.
            for val in _iter_command_values(tool_input, cmd_keys):
                findings.extend(analyze_command(val, protected, protected_pkgs))

        if tool_name in _WRITE_TOOLS:
            path = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
            base = os.path.basename(path).lower()
            low_path = path.lower()
            if (
                base in secret_bases
                or base.startswith(".env.")
                or base.endswith(".env")  # .env, prod.env, production.env, ...
                or low_path.endswith(secret_sfx)
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
                seg in _norm_path(low_path) or seg in _deobf_path(low_path) for seg in protected
            ) and _resolves_into_protected(path, protected):
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
        if tool_name != "Bash" and tl not in readonly:
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
                if any(seg in _norm_path(low) or seg in _deobf_path(low) for seg in protected) or (
                    path_like and _resolves_into_protected(s, protected)
                ):
                    findings.append(
                        Finding.fail(
                            "self_protection",
                            severity="CRITICAL",
                            message=f"refusing a `{tool_name}` call that targets a protected "
                            f"control path (gate config, hook, the recusal enforcement "
                            f"package, or git hooks): {s}",
                            path=s,
                        )
                    )
                    break

        return findings

    return policy
