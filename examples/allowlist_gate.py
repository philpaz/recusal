"""
Allowlist mode (default-deny), the stronger posture, as a runnable gate.

Every *deny-list* gate shares one ceiling: it cannot catch a command whose name is built
at runtime. `c=$'\\x72\\x6d'; $c -rf /` never contains the literal string `rm`, so no
substring or regex matcher ever sees it. The robust answer inverts the default: **deny
everything, allow only what you affirmatively vet.** For `Bash` that means rejecting shell
metacharacters (so a command can't chain, substitute, or expand into something else) and
requiring a vetted first binary. That defeats exactly the runtime-construction bypasses a
deny-list cannot.

This file is both a demo and a real policy:

    python examples/allowlist_gate.py      # prints the deny-list-vs-allowlist comparison

    # or wire it as a Claude Code PreToolUse hook (absolute path in .claude/settings.json):
    #   { "type": "command", "command": "python3 .../examples/allowlist_gate.py --hook" }
    # The --hook flag runs the gate against a real PreToolUse event instead of the demo.

The trade-off is honest: an allowlist is stricter and needs maintenance (you add
capabilities as the agent legitimately needs them), but it fails *toward* refusal instead
of away from it. This is a reference policy - read it, tune the lists to your system.
"""

import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recusal import Finding  # noqa: E402
from recusal.claude_code import decide, run_pretooluse_hook  # noqa: E402

WORKSPACE = os.path.abspath("./workspace")
# Binaries safe regardless of arguments. Tools that can mutate (git, sed, find, rm) are NOT
# here; allowlist those only with explicit per-subcommand rules of your own.
SAFE_BINARIES = {
    "ls", "cat", "head", "tail", "grep", "rg", "wc", "pwd", "stat", "diff",
    "pytest", "ruff", "mypy",
}  # fmt: skip
SHELL_META = set(";|&`$<>(){}\n\\")  # chaining / substitution / redirection / expansion


def _under(root: str, path: str) -> bool:
    try:
        return os.path.commonpath([root, os.path.abspath(path)]) == root
    except ValueError:  # different drives on Windows
        return False


def _bash_ok(cmd: str) -> bool:
    if set(cmd) & SHELL_META:  # can't reason about an expanded command -> refuse
        return False
    try:
        argv = shlex.split(cmd)
    except ValueError:  # unbalanced quotes -> refuse
        return False
    return bool(argv) and argv[0] in SAFE_BINARIES


# Each entry affirmatively vets a call. Anything not covered here is refused by default.
ALLOW = {
    "Read": lambda i: True,
    "Grep": lambda i: True,
    "Glob": lambda i: True,
    "Bash": lambda i: _bash_ok(str(i.get("command", ""))),
    "Write": lambda i: _under(WORKSPACE, str(i.get("file_path", ""))),
    "Edit": lambda i: _under(WORKSPACE, str(i.get("file_path", ""))),
}


def policy(tool_name: str, tool_input: dict) -> list:
    """Default-deny: return no findings (defer) only for an affirmatively vetted call;
    refuse everything else with a CRITICAL finding."""
    check = ALLOW.get(tool_name)
    if check and check(tool_input):
        return []  # affirmatively allowed -> defer to Claude Code's normal flow
    return [
        Finding.fail(
            "not_allowlisted",
            severity="CRITICAL",
            message=f"{tool_name} call is not on the allowlist",
        )
    ]


# --- a naive deny-list, only to show the ceiling this example clears --------------------


def _denylist_policy(tool_name: str, tool_input: dict) -> list:
    """A *good* deny-list: it even de-obfuscates (strips quotes/backticks/backslashes) before
    matching, like the real dogfood hook. It still cannot see a command name assembled at
    runtime - that is the ceiling, not a weakness of this particular list."""
    if tool_name != "Bash":
        return []
    cmd = str(tool_input.get("command", ""))
    deobf = cmd.replace("'", "").replace('"', "").replace("`", "").replace("\\", "")
    if "rm -rf" in cmd or "rm -rf" in deobf:
        return [Finding.fail("destructive", severity="CRITICAL", message="rm -rf (de-obfuscated)")]
    return []


# Payloads that all *intend* to delete a tree. A deny-list sees only the first two.
_ATTACKS = [
    ("literal rm -rf", "rm -rf /repo"),
    ("obfuscated (quotes)", "'r''m' -rf /repo"),
    ("hex-built name", r"c=$'\x72\x6d'; $c -rf /repo"),
    ("char-built name", "a=r;b=m;${a}${b} -rf /repo"),
    ("eval of base64", "eval $(echo cm0gLXJmIC9yZXBv | base64 -d)"),
]
_SAFE = ("run pytest", "pytest -q")


def main() -> None:
    print("RECUSAL - deny-list vs allowlist (offline)\n")
    print("  Same destructive INTENT, five encodings. A deny-list (even one that")
    print("  de-obfuscates) cannot see a name built at runtime; the allowlist refuses")
    print("  anything it did not affirmatively vet.\n")
    print(f"  {'attack':<22}{'deny-list':<14}allowlist")
    print("  " + "-" * 48)
    for label, cmd in _ATTACKS:
        deny = decide("Bash", {"command": cmd}, _denylist_policy)[0]
        allow = decide("Bash", {"command": cmd}, policy)[0]
        print(f"  {label:<22}{deny.upper():<14}{allow.upper()}")
    safe = decide("Bash", {"command": _SAFE[1]}, policy)[0]
    print(f"  {_SAFE[0]:<22}{'-':<14}{safe.upper()}  (vetted binary -> still usable)")
    print(
        "\n  DEFER = the gate had no opinion (the call proceeds); DENY = refused.\n"
        "  Even a de-obfuscating deny-list DEFERS the three runtime-constructed names. The\n"
        "  allowlist DENIES every unvetted call and still lets the vetted `pytest` through.\n"
        "  That is the ceiling a deny-list cannot clear, and why high-stakes tools want\n"
        "  default-deny."
    )


if __name__ == "__main__":
    # `python examples/allowlist_gate.py`         -> the demo (default)
    # `python examples/allowlist_gate.py --hook`  -> act as a real Claude Code PreToolUse hook
    if "--hook" in sys.argv:
        run_pretooluse_hook(policy)
    else:
        main()
