# Command line reference

Every command and flag of `python -m recusal` (also installed as `recusal`), as
`--help` prints them. `tests/test_cli_reference.py` reads the parser and fails if a
flag is missing here, so this page cannot fall behind the code.

```text
python -m recusal [--version] {init,verdict,audit,mcp,doctor,demo} ...
```

| Flag | What it does |
|---|---|
| `--version` | Prints the installed version. |

## `init`

Scaffolds a Claude Code PreToolUse gate: `.claude/hooks` and `settings.json`.

| Flag | What it does |
|---|---|
| `--dir DIR` | The project directory to scaffold into. Default: the current directory. |
| `--posture {deny-list,allowlist}` | `deny-list` refuses what is known to be bad and defers the rest (broad channels). `allowlist` denies by default: nothing runs unless it is named (high-stakes channels). |
| `--writable-root WRITABLE_ROOT` | Allowlist posture only: the directory subtree that writes are confined to. |
| `--launcher {auto,posix,powershell,both}` | Which fail-closed launcher to register. `auto` (the default) is PowerShell on Windows and POSIX elsewhere; `both` suits a `settings.json` shared across operating systems, at the cost of the gate running twice per call on Windows hosts with Git Bash. |
| `--repair-launcher` | Replaces the canonical recusal launchers already in `settings.json` with the right ones for this host: the migration path for a Windows install from before 0.4.2, whose POSIX launcher fails open under the PowerShell fallback. Custom hooks and the gate's policy file are never touched. |

## `verdict`

Adjudicates a findings JSON file to PASS, RETRY or FAIL (exit 0, 1 or 2): the CI
primitive. Its argument is a JSON array of findings, or an object with a `findings`
array; `-` reads standard input.

| Flag | What it does |
|---|---|
| `--json` | Writes the verdict as JSON instead of text. |
| `--lenient` | Accepts findings that omit `status` or `passed` as passing. The strict default rejects them, so a failure cannot pass silently. |

## `audit verify`

Verifies a JSONL audit log's hash chain: exit 0 intact, 1 broken, 2 error. Its
argument is the path to the log.

| Flag | What it does |
|---|---|
| `--expect-head COUNT:HASH` | An anchor for the head kept outside the log; catches truncation, a rewritten tail and forged appends. |
| `--json` | Writes the result as JSON instead of text. |

## `mcp pin` and `mcp verify`

`mcp pin` pins the observed catalog to a deterministic manifest: exit 0 pinned or
nothing to do, 1 when tool declarations or server instructions need review, 2 refused.
`mcp verify` checks the observed sources, server instructions and tool declarations
against the pinned manifest: exit 0 match, 2 drift, unpinned, unobserved or error.
[MCP](MCP.md) explains both in full.

Both commands take these flags:

| Flag | What it does |
|---|---|
| `--from FILE` | A JSON observation, the way in for HTTP servers. The rich shape, `{server: {"instructions": str or null, "tools": [...]}}`, pins instructions too. The legacy shapes (a `{server: [tools]}` mapping, or one server's raw `tools/list` result with `--server`) do not establish server-instruction coverage and are recorded as `observed: false`. |
| `--server SERVER` | The server's name when `--from` holds one server's observation. |
| `--stdio NAME COMMAND` | Observes a stdio MCP server live by executing this command, for example `--stdio github 'npx -y @modelcontextprotocol/server-github'`. Repeatable. |
| `--claude-config PATH` | Observes every stdio server in a Claude Code `.mcp.json` by executing its declared commands. Treat the file as executable code and review it first. |
| `--timeout TIMEOUT` | Seconds to wait for each stdio server request. Default 30. |
| `--minimal-env` | The default since 0.5.0, kept for compatibility: stdio servers start with a minimal environment (`PATH` and the like, plus the config's own `env`) instead of the full shell environment. |
| `--inherit-env` | Opts out of the minimal environment and hands stdio servers the full parent environment, as Claude Code launches them. A server being pinned is not yet trusted with your API keys, so this is an explicit, named trade. |
| `--json` | Writes the result as JSON instead of text. |

`mcp pin` only:

| Flag | What it does |
|---|---|
| `--out OUT` | The manifest to write. Default: `mcp-manifest.json`. |
| `--approve-server-launch` | Required with `--stdio` or `--claude-config`: records that a person reviewed the configured commands and approves executing them to observe their catalogs. `verify` then compares each launch spec before launching. |
| `--update` | Allows replacing an existing manifest whose content differs. A pin is approved truth, so replacing it is a deliberate step. |
| `--claude-plugin NAME` | Declares this server key as a Claude plugin runtime segment (repeatable): callable names are derived by Claude's documented normalization, and two raw names that normalize to one callable refuse the pin. Never inferred from the key's spelling. |
| `--resolve-executable` | Also pins the path and sha256 of the file each launched stdio command resolves to, so `verify` refuses a swapped executable behind an unchanged command. Requires `--stdio` or `--claude-config`. Covers the first process image only. |
| `--scope LABEL` | A label for what this pin observed (a project config, a machine, an environment), stored verbatim and shown by `verify`. A changed scope under `--update` is a review warning, never a refusal. The label's truth is never verified. |
| `--force` | Pins even when the screen flags tool declarations, server instructions or source configuration, recording that a person reviewed and accepted them. |

`mcp verify` only:

| Flag | What it does |
|---|---|
| `--manifest MANIFEST` | The pinned manifest to verify against. Default: `mcp-manifest.json`. |
| `--removed NAME` | Acknowledges that this pinned server was deliberately removed (repeatable). Without it, a pinned server absent from the whole observation refuses. To remove every MCP server, remove or replace the manifest itself. |

## `doctor`

Health-checks a scaffolded gate: exit 0 healthy, 1 degraded, 2 not installed.

| Flag | What it does |
|---|---|
| `--dir DIR` | The project directory to inspect. Default: the current directory. |
| `--json` | Writes the result as JSON instead of text. |

## `demo`

Shows the gate refusing, offline. Exit 0 means the demo ran; it is not a verdict.

| Flag | What it does |
|---|---|
| `--scenario {all,wrong-subject,destructive-shell,mcp-drift,expired-authorization}` | Which scenario to run. Default: all four, in order. |
| `--list` | Lists the scenarios and what each demonstrates. |
