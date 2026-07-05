"""
Example Claude Code PreToolUse hook backed by Recusal.

Register it in ``.claude/settings.json``. Use the interpreter-probe launcher (not a bare
``python3 ...``): a hook whose command fails to launch is a *non-blocking* error in Claude
Code, so a missing ``python3`` would silently disable the gate; the loop runs the first
``python3``/``python``/``py`` that is >=3.9 and coerces any nonzero exit into ``exit 2``
(the blocking code), so it fails **closed**:

    {
      "hooks": {
        "PreToolUse": [
          { "matcher": ".*", "hooks": [
            { "type": "command",
              "command": "for p in python3 python py; do \"$p\" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null && { \"$p\" \"$CLAUDE_PROJECT_DIR/.claude/hooks/claude_code_gate.py\"; rc=$?; [ \"$rc\" = 0 ] || { echo 'gate: hook did not run cleanly; failing closed' >&2; exit 2; }; exit 0; }; done; echo 'gate: no python>=3.9; failing closed' >&2; exit 2" }
          ]}
        ]
      }
    }

Then Recusal refuses destructive bash and writes to secret files *before* Claude
Code runs them, even in bypassPermissions / auto mode. Anything the policy has no
opinion on defers to Claude Code's normal permission flow.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recusal import Finding  # noqa: E402
from recusal.claude_code import run_pretooluse_hook  # noqa: E402


def policy(tool_name: str, tool_input: dict) -> list:
    findings = []
    if tool_name == "Bash":
        cmd = str(tool_input.get("command", ""))
        if "rm -rf" in cmd or ":(){:|:&};:" in cmd:
            findings.append(
                Finding.fail(
                    "destructive_bash",
                    severity="CRITICAL",
                    message=f"refusing destructive shell command: {cmd!r}",
                    command=cmd,
                )
            )
    if tool_name in ("Write", "Edit", "MultiEdit"):
        path = str(tool_input.get("file_path", ""))
        base = os.path.basename(path)
        # A file_path guard can't see a Bash redirect (echo X > .env); gate Bash for that.
        if base == ".env" or base.startswith(".env.") or path.endswith((".pem", ".key")):
            findings.append(
                Finding.fail(
                    "secret_write",
                    severity="CRITICAL",
                    message=f"refusing write to a secret file: {path}",
                    path=path,
                )
            )
    return findings


if __name__ == "__main__":
    run_pretooluse_hook(policy)
