"""
Example Claude Code PreToolUse hook backed by Recusal.

Register it in ``.claude/settings.json`` (use an absolute path):

    {
      "hooks": {
        "PreToolUse": [
          { "matcher": ".*", "hooks": [
            { "type": "command",
              "command": "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/claude_code_gate.py" }
          ]}
        ]
      }
    }

Then Recusal refuses destructive bash and writes to secret files *before* Claude
Code runs them — even in bypassPermissions / auto mode. Anything the policy has no
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
            findings.append(Finding.fail(
                "destructive_bash", severity="CRITICAL",
                message=f"refusing destructive shell command: {cmd!r}", command=cmd))
    if tool_name in ("Write", "Edit"):
        path = str(tool_input.get("file_path", ""))
        if "/.env" in path or path.endswith((".pem", ".key")):
            findings.append(Finding.fail(
                "secret_write", severity="CRITICAL",
                message=f"refusing write to a secret file: {path}", path=path))
    return findings


if __name__ == "__main__":
    run_pretooluse_hook(policy)
