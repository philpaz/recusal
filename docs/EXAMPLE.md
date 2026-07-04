# Worked example: a database-admin agent you can leave in auto mode

One complete, copy-paste configuration for a single real use case, so you can see exactly
what adopting Recusal looks like end to end.

## The use case

You run a Claude Code agent in **auto mode** (`bypassPermissions`) to handle routine ops:
query and fix data in a staging database, edit config, run scripts. It is useful precisely
because nobody watches each step. The risk is that one bad tool call does real damage before
anyone looks:

- `rm -rf` outside the working directory
- a `DELETE` or `UPDATE` run through `psql` with no `WHERE` clause
- a write to a secrets file (`.env`, a private key)
- `curl ... | sh` (a pipe-to-shell install)

You want the agent to stay autonomous but be unable to take those specific actions. That is
the job of a Recusal `PreToolUse` hook: it adds refusals for the actions you have named as
off limits, and it stays silent (defers) on everything else, so Claude Code's normal
permission flow is unchanged. A `deny` holds even under `bypassPermissions`, so the agent
cannot switch its own mode to get around it.

## 1. Install

```bash
pip install recusal
```

## 2. Write the gate

`.claude/hooks/gate.py` (the whole policy for this use case, composed from a few rules):

```python
import os
from recusal import Finding
from recusal.claude_code import run_pretooluse_hook

WORKSPACE = os.path.abspath("./workspace")
PROTECTED = (".env", ".pem", ".key", "id_rsa", "credentials", "secrets")
SHELL_RED = ("rm -rf", "rm -fr", "mkfs", "dd if=", "chmod -r 777")

def policy(tool_name, tool_input):
    findings = []

    if tool_name == "Bash":
        cmd = tool_input.get("command", "").lower()
        if any(m in cmd for m in SHELL_RED):
            findings.append(Finding.fail("destructive_shell", severity="CRITICAL",
                                         message="refusing a destructive shell command"))
        if "curl" in cmd and ("| sh" in cmd or "| bash" in cmd):
            findings.append(Finding.fail("pipe_to_shell", severity="CRITICAL",
                                         message="refusing curl piped into a shell"))
        if any(client in cmd for client in ("psql", "mysql", "sqlite3")):
            if ("delete" in cmd or "update" in cmd) and "where" not in cmd:
                findings.append(Finding.fail("sql_scope", severity="CRITICAL",
                                             message="refusing destructive SQL with no WHERE clause"))
            if "drop table" in cmd or "truncate" in cmd:
                findings.append(Finding.fail("sql_schema", severity="CRITICAL",
                                             message="refusing DROP or TRUNCATE on a table"))

    if tool_name in ("Write", "Edit", "MultiEdit"):
        path = tool_input.get("file_path", "")
        ap = os.path.abspath(path)
        try:  # commonpath, not startswith ("/workspace_evil" would slip past startswith)
            inside = os.path.commonpath([WORKSPACE, ap]) == WORKSPACE
        except ValueError:  # different drives on Windows
            inside = False
        if any(p in path.lower() for p in PROTECTED):
            findings.append(Finding.fail("protected_file", severity="CRITICAL",
                                         message=f"refusing a write to a secret file: {path}"))
        elif not inside:
            findings.append(Finding.fail("path_confinement", severity="CRITICAL",
                                         message=f"refusing a write outside the workspace: {path}"))

    return findings   # empty list means defer to Claude Code's normal flow

run_pretooluse_hook(policy)
```

## 3. Register it

`.claude/settings.json` (use an absolute path to the hook):

```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": ".*", "hooks": [
        { "type": "command",
          "command": "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/gate.py" }
      ]}
    ]
  }
}
```

That is the entire setup. Three files touched, nothing running in the background, no service.

## 4. What the agent experiences

The agent runs as usual. When it proposes one of the named actions, Recusal denies it before
the tool runs and hands the agent the reason, which it reads and re-plans around:

```text
agent proposes:  Bash  rm -rf /var/data
  -> DENY   Recusal refused `Bash` [FAIL]: refusing a destructive shell command

agent proposes:  Bash  psql -c "DELETE FROM sessions"
  -> DENY   Recusal refused `Bash` [FAIL]: refusing destructive SQL with no WHERE clause

agent proposes:  Write  ./.env  (DB_PASSWORD=...)
  -> DENY   Recusal refused `Write` [FAIL]: refusing a write to a secret file: ./.env

agent proposes:  Edit  ./workspace/app/config.py
  -> (no opinion) defer to Claude Code's normal flow, the edit proceeds
```

The deny is honored even though the agent is running in `bypassPermissions`. The clean edit
is not blocked: the gate only ever adds refusals, it never strips Claude Code's own prompts.

## 5. Keep the receipts (optional)

To record every decision in a tamper-evident, hash-chained log for later review, pair the
gate with `recusal.audit` (see [`../examples/audit_demo.py`](../examples/audit_demo.py)).
The record maps cleanly onto agent logging and audit requirements.

## What you configured

- An agent that stays fully autonomous, minus four classes of action you decided it must
  never take.
- A refusal that is deterministic and replayable: the same proposed call gets the same
  answer every time, including the `no`, with a reason you can read and audit.
- No model in the decision path, and no change to how the agent works when it behaves.

One honest caveat: this gate is a **deny-list**, it stops the four named classes and the
common variants, not everything a determined agent could construct (`python script.py`
runs code no string match reads). If the stakes warrant "nothing runs unless named," flip
the same hook to the shipped allowlist mode,
`run_pretooluse_hook(allowlist_policy(writable_root="./workspace"))`, see
[HOWTO §1](HOWTO.md) "Two postures, two claims."

Want different rules? Lift more from the [policy cookbook](COOKBOOK.md), or read
[HOWTO](HOWTO.md) for the Agent SDK and any-loop surfaces.
