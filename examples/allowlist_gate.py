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

    # or wire it as a Claude Code PreToolUse hook (absolute path in .claude/settings.json).
    # Use the interpreter-probe launcher, not a bare python3, so a missing interpreter fails
    # CLOSED (a hook that can't launch is a non-blocking error in Claude Code -> fail open):
    #   { "type": "command",
    #     "command": "for p in python3 python py; do \"$p\" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null && { \"$p\" .../examples/allowlist_gate.py --hook; rc=$?; [ \"$rc\" = 0 ] || exit 2; exit 0; }; done; exit 2" }
    # The --hook flag runs the gate against a real PreToolUse event instead of the demo.

The trade-off is honest: an allowlist is stricter and needs maintenance (you add
capabilities as the agent legitimately needs them), but it fails *toward* refusal instead
of away from it. The policy ships as library API, ``recusal.claude_code.allowlist_policy``;
this file wires it up and shows *why* it clears the deny-list ceiling, tune the lists to
your system.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recusal import Finding  # noqa: E402
from recusal.claude_code import allowlist_policy, decide, run_pretooluse_hook  # noqa: E402

WORKSPACE = os.path.abspath("./workspace")

# Default-deny, straight from the library: Read/Grep/Glob defer; `Bash` needs a vetted
# first binary (ls, cat, grep, diff, ..., read/inspect tools only; interpreters AND
# code-through-an-argument tools like pytest/mypy/rg are deliberately unlisted, so
# `python script.py` and `pytest` are refused: each runs a program the gate never reads)
# and no shell metacharacters; writes stay under WORKSPACE; every other tool is refused.
policy = allowlist_policy(writable_root=WORKSPACE)


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
_SAFE = ("list a directory", "ls -la src")
# pytest is NOT vetted: it imports conftest.py / the test files it collects, i.e. it runs
# arbitrary code through an argument -- exactly the write-a-script-then-run-it bypass this
# posture closes -- so the allowlist refuses it (shown as its own row below).
_SAFE_REFUSED = ("run pytest (code-exec)", "pytest -q")


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
    print(f"  {_SAFE[0]:<22}{'-':<14}{safe.upper()}  (vetted read -> still usable)")
    refused = decide("Bash", {"command": _SAFE_REFUSED[1]}, policy)[0]
    print(f"  {_SAFE_REFUSED[0]:<22}{'-':<14}{refused.upper()}  (code-exec binary -> refused)")
    print(
        "\n  DEFER = the gate had no opinion (the call proceeds); DENY = refused.\n"
        "  Even a de-obfuscating deny-list DEFERS the three runtime-constructed names. The\n"
        "  allowlist DENIES every unvetted call, lets a vetted read like `ls` through, and\n"
        "  still refuses `pytest` (it runs code via conftest.py -> not arg-safe). That is the\n"
        "  ceiling a deny-list cannot clear, and why high-stakes tools want default-deny."
    )


if __name__ == "__main__":
    # `python examples/allowlist_gate.py`         -> the demo (default)
    # `python examples/allowlist_gate.py --hook`  -> act as a real Claude Code PreToolUse hook
    if "--hook" in sys.argv:
        run_pretooluse_hook(policy)
    else:
        main()
