# Recusal Gate (Claude Code plugin)

**This plugin denies tool calls.** It registers a `PreToolUse` hook that checks every
tool call Claude Code proposes, in every project, and refuses the dangerous ones before
they run. A refusal holds even under `bypassPermissions`. No model makes the decision:
the shipped deny list does.

## What it refuses

- Destructive shell commands, such as `rm -rf`.
- Git commands that throw away work or history, such as `git reset --hard`,
  `git branch -D` or `git push --delete`.
- Writes to secret files, such as `.env`.
- Writes to the controls that govern the session: `.claude/settings.json`, `.mcp.json`
  and the MCP manifest.
- Uninstalling, reinstalling or shadowing the `recusal` package itself.

Anything else defers: Claude Code's own permission prompts still apply. The plugin adds
refusals and never removes one.

## Install

```bash
claude plugin marketplace add philpaz/recusal
claude plugin install recusal-gate@recusal
```

No `pip install` is needed. The plugin carries its own copy of the Recusal runtime
(`vendor/recusal`), and it refuses every tool call rather than run with a missing,
substituted or mismatched copy.

## Limits

- The deny list matches text, so a command can be disguised past it, and `python
  script.py` runs code no string check reads. For a narrow, high-stakes channel, use
  `python -m recusal init --posture allowlist` instead, which refuses everything not
  named.
- The launcher is POSIX: macOS, Linux, and Windows with Git Bash. On Windows without Git
  Bash, use `python -m recusal init`, which registers a PowerShell launcher.
- Claude Code lets a hook that times out through (600 seconds by default); the shipped
  policy decides in milliseconds.

To tailor the policy to one project, scaffold a project gate with `python -m recusal
init` and edit it. Full documentation, the security model and how to verify a release:
<https://github.com/philpaz/recusal>.

## License

Apache-2.0, the same as the repository ([`LICENSE`](../LICENSE)).
