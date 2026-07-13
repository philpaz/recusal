# Proven, Recusal governs its own repository

Recusal isn't only shown on toy inputs. It is built to govern the development of *this*
repository. A real Claude Code `PreToolUse` hook, [`.claude/hooks/recusal_gate.py`](../.claude/hooks/recusal_gate.py),
ships with the repo and, once registered (a deliberate step, see below), adjudicates every
tool call first, so the dangerous ones the policy recognizes, destructive shell commands,
writes to secret/protected files, and edits that would disable the gate itself, are
**refused before they run**, even under `bypassPermissions`. Every refusal below is
reproducible by anyone, registered or not, by piping the exact payload into the hook.

The governance library is the thing governing its own development. That is the dog food.

## What's installed

- [`.claude/hooks/recusal_gate.py`](../.claude/hooks/recusal_gate.py): the hook, a thin shim
  that wires the repo-protection policy from `recusal.deny_list` (`deny_list_policy()`, the
  installable, unit-tested engine) into `PreToolUse`. That policy refuses `rm -rf` and its
  flag variants, force-push, `reset --hard`, `curl … | sh`, writes to `.env` / `*.pem` /
  `*.key` and shell redirects to them, and edits to the gate's own settings/hook so an agent
  can't disable it. It is a substring/regex
  deny-list, a *baseline*, not a guarantee: an obfuscated command can evade any literal
  matcher, and for a narrow high-stakes channel the refuse-by-default allowlist path fits
  better (neither is "better" in the abstract; the channel decides). What this proves is the
  seam, not an exhaustive list.
- Registration, see [`.claude/settings.json.example`](../.claude/settings.json.example).
  Installing a permission-changing hook is a **deliberate, reviewed step**: Claude Code's
  own auto mode will (correctly) ask you to confirm it. That confirmation *is* the
  separation-of-powers point, a control that changes what an agent may do should not be
  installed silently.

## It refuses, for real

Each block below is the **verbatim** output of piping the exact JSON Claude Code sends on
`PreToolUse` into the installed hook (the path in a reason simply echoes the path your
payload carried). Reproduce any line yourself:

```bash
echo '<payload>' | python .claude/hooks/recusal_gate.py   # or python3 / py, whichever your OS has
```

**`rm -rf`** (the recursive-`rm` guard fires on `-r` in any flag order, force or not)
```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Recusal refused `Bash` [FAIL]: refusing destructive command (rm -r)"}}
```

**`git push --force`**
```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Recusal refused `Bash` [FAIL]: refusing destructive command (git push --force)"}}
```

**`curl … | sh`** (and `… | python`/`perl`/`ruby`/`node`, same attack class)
```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Recusal refused `Bash` [FAIL]: refusing to pipe output straight into a shell/interpreter"}}
```

**Write to `.env`**
```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Recusal refused `Write` [FAIL]: refusing write to a protected/secret file: .env"}}
```

**Edit the gate's own config (self-protection)**
```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Recusal refused `Edit` [FAIL]: refusing a `Edit` call that targets a protected control path (gate config/hook or git hooks): .claude/settings.json"}}
```

**Delete the hook itself (self-protection covers removal, not just edits)**, `rm .claude/hooks/recusal_gate.py`
```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Recusal refused `Bash` [FAIL]: refusing a command that edits or removes a protected control path (gate config/hook or git hooks)"}}
```

A clean call, a `Read`, or `Bash` running `pytest -q`, produces **no output**: the hook
*defers* to Claude Code's normal permission flow. The gate only ever adds refusals; it never
strips your existing prompts.

## Locked by CI

[`tests/test_dogfood.py`](../tests/test_dogfood.py) loads this exact policy and asserts the
refusals (and the defers), so the proof cannot silently rot; the CI workflow is configured to run these regression tests.

## A reference architecture: catching a real bug

Recusal also ships as a **reference integration** showing how to wire it into an agent: a
FastAPI agent (a relationship agent with confirm-gated CRM writes) where a subject-match
guard closes a named audit finding, **wrong-subject write**. The agent could stage a write
against a *different* member than the conversation was about, and the human confirm card
would wave it through.

The guard sits at the confirm endpoint and adjudicates the write with Recusal's
`compute_verdict`: the write must target the member who was active when it was proposed.
Output from the wired guard:

```text
wrong subject  -> write targets C-9988 but the active member this turn is C1001   (refused)
right subject  -> None                                                            (allowed)
no active mbr  -> None                                                            (fail-open)
```

On a mismatch the confirm endpoint returns `409 Refused by subject guard …` and the CRM
write never executes. This is a reference pattern, not a deployment claim: it lives in a
separate private codebase and is not reproducible from this repo, and the integration is
intentionally fail-open (Recusal is an optional dependency), so it can never block a
previously-working flow. What it demonstrates is the pattern: a deterministic, independent
guard catching a real, named wrong-subject bug before the write runs.

## Release proofs on the record (v0.5.12, 2026-07-13)

For v0.5.12 (commit `f02d37d`, widening the package-manager matcher and making the
protected-name contracts explicit):

- **Workflow evidence is public**: CI run [29267866649](https://github.com/philpaz/recusal/actions/runs/29267866649)
  (all 10 jobs green) and release run [29269214665](https://github.com/philpaz/recusal/actions/runs/29269214665)
  (full suite at the release commit, hash-locked install, `--no-isolation` build,
  neutral-directory wheel check, Trusted Publishing).
- **A form that deferred in 0.5.11, proven refused from the published wheel** (fresh
  venv, `pip install recusal==0.5.12` from PyPI, exact payload piped into the repo
  hook), `pip --python .venv uninstall recusal`:

```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Recusal refused `Bash` [FAIL]: refusing a package-manager command that uninstalls, reinstalls, or shadows the gate's enforcement package (recusal); manage it outside the governed session"}}
```

- **The contracts hold from the same wheel**: `protected_packages=("My.Gate",)`
  refuses `pip uninstall my_gate` / `my-gate` / `MY.GATE` (canonical PEP 503
  identity), `protected_packages=("",)` raises `ValueError` at construction, and the
  negative space is unchanged (`pip install requests`, `uv add httpx`,
  `pip show recusal`, `uv tool install some-tool` all defer). Pinned as 98
  deterministic tests in
  [`tests/test_deny_list_package_protection.py`](../tests/test_deny_list_package_protection.py),
  including the 66 shipped at v0.5.11, which pass unchanged against the widened
  matcher.

## Release proofs on the record (v0.5.11, 2026-07-13)

For v0.5.11 (commit `5d1acc1`, package-manager self-protection and the capability-first
README):

- **Workflow evidence is public**: CI run [29225000959](https://github.com/philpaz/recusal/actions/runs/29225000959)
  (all 10 jobs green) and release run [29225128002](https://github.com/philpaz/recusal/actions/runs/29225128002)
  (full suite at the release commit, hash-locked install, `--no-isolation` build,
  neutral-directory wheel check, Trusted Publishing).
- **Proven from the published wheel**: each block below is the verbatim output of piping
  the exact `PreToolUse` payload into the repo hook using a fresh venv's Python with
  `pip install recusal==0.5.11` from PyPI (the hook appends the repo path *last*, so the
  installed distribution adjudicates, not the working tree).

**The exact reported gap**, `pip uninstall recusal` (deferred in 0.5.10 and earlier):
```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Recusal refused `Bash` [FAIL]: refusing a package-manager command that uninstalls, reinstalls, or shadows the gate's enforcement package (recusal); manage it outside the governed session"}}
```

**An obfuscated spelling**, `pip${IFS}uninstall${IFS}recusal` (same de-obfuscation as
every other check):
```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Recusal refused `Bash` [FAIL]: refusing a package-manager command that uninstalls, reinstalls, or shadows the gate's enforcement package (recusal); manage it outside the governed session"}}
```

**A non-Bash tool carrying a command-like key** (an MCP shell cannot be a second,
ungated shell), payload `{"tool_name": "mcp__runner__execute", "tool_input": {"args":
{"Command": "pip uninstall recusal"}}}`:
```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Recusal refused `mcp__runner__execute` [FAIL]: refusing a package-manager command that uninstalls, reinstalls, or shadows the gate's enforcement package (recusal); manage it outside the governed session"}}
```

**The negative space holds**: `pip install requests` produces **no output**, the hook
defers and the call proceeds to the normal permission flow.

- The same properties are pinned as 66 deterministic tests in
  [`tests/test_deny_list_package_protection.py`](../tests/test_deny_list_package_protection.py)
  at the 0.5.11 tag: positive, obfuscated, command-key, and negative cases (mutating
  other packages and read-only pip subcommands against `recusal` still defer). The
  named ceiling is documented, not papered over: an install that provides the package
  without naming it (`pip install -e .`, `-r requirements.txt`) is unreadable to a
  string matcher; the pinned, write-protected venv stays the real defense.

## Release proofs on the record (v0.5.10, 2026-07-12)

For v0.5.10 (commit `65b1b27`, publishing the loader-wide collision invariant):

- **Workflow evidence is public**: CI run [29215512313](https://github.com/philpaz/recusal/actions/runs/29215512313)
  (all 10 jobs green) and release run [29215621648](https://github.com/philpaz/recusal/actions/runs/29215621648)
  (full suite at the release commit, hash-locked install, `--no-isolation` build,
  neutral-directory wheel check, Trusted Publishing).
- **The hand-edited collision refusal, proven from the published wheel** (a manifest
  whose pins are each individually canonical but whose raw tools collide on one
  callable): `load_manifest` refuses with "ambiguous callable identity", and
  `manifest_policy` fails closed on the artifact (`mcp_manifest_unavailable`).
  Pinned as three deterministic tests in
  [`tests/test_mcp_runtime_identity.py`](../tests/test_mcp_runtime_identity.py).

## Release proofs on the record (v0.5.9, 2026-07-12)

For v0.5.9 (commit `bc9a722`, manifest v6 runtime identity):

- **Workflow evidence is public**: CI run [29213909534](https://github.com/philpaz/recusal/actions/runs/29213909534)
  (all 10 jobs green) and release run [29213965813](https://github.com/philpaz/recusal/actions/runs/29213965813)
  (full suite at the release commit, hash-locked install, `--no-isolation` build,
  neutral-directory wheel check, Trusted Publishing).
- **A spec-valid dotted plugin tool pins with both identities** (verbatim, CLI with
  `--claude-plugin`):

```text
runtime: {'mode': 'claude_plugin'}
raw key -> callable: {'admin.tools.list': 'admin_tools_list', 'query': 'query'}
```

- **`PreToolUse` membership uses the callable identity** (verbatim, library):

```text
normalized callable: AUTHORIZED
raw dotted spelling: mcp_not_pinned -> REFUSED
```

- **A callable collision refuses the pin** (exit 2, no manifest written):

```text
FAIL (failed closed) - catalog cannot be pinned: ... raw tools 'admin.tools.list' and 'admin_tools_list' both normalize to the Claude callable 'admin_tools_list'; ambiguous callable identity cannot be pinned
```

- **A v5 manifest refuses with the migration message** (exit 2):

```text
FAIL (failed closed) - no usable manifest ... manifest_version 5 predates runtime-identity modeling ... re-pin with `recusal mcp pin` to record explicit runtime naming modes
```

- `recusal mcp verify` needs no flag: the mode lives in the manifest (verified PASS
  against the same dump). The same properties are pinned as 25 deterministic tests in
  [`tests/test_mcp_runtime_identity.py`](../tests/test_mcp_runtime_identity.py),
  including digest/audit-provenance coverage of runtime identity and the
  `list_changed` raw-swap regression.

## Release proofs on the record (v0.5.8, 2026-07-12)

For v0.5.8 (commit `cb61e9e`, a claim-correction release):

- **Workflow evidence is public**: CI run [29211767659](https://github.com/philpaz/recusal/actions/runs/29211767659)
  (all 10 jobs green) and release run [29211816252](https://github.com/philpaz/recusal/actions/runs/29211816252)
  (full suite at the release commit, hash-locked install, `--no-isolation` build,
  neutral-directory wheel check, Trusted Publishing).
- **The correction itself is the record**: 0.5.7 published that Claude does not
  document plugin callable-name normalization; the rule is explicitly documented,
  the error was this repository's verification sweep, and the claim (plus its
  downstream "false denial only" and "observe and pin" statements) is amended in
  place in the README, the 0.5.7 changelog entry, and the published 0.5.7 release
  notes rather than silently rewritten.
- **The alias residual, pinned as a deterministic demonstration**
  ([`tests/test_mcp_observation_decommission.py`](../tests/test_mcp_observation_decommission.py)):
  the approved callable spelling passes at call time, and the next verify refuses
  the swapped raw declaration as unpinned capability plus a removed pinned tool.

## Release proofs on the record (v0.5.7, 2026-07-12)

For v0.5.7 (commit `a7fd431`):

- **Workflow evidence is public**: CI run [29210780582](https://github.com/philpaz/recusal/actions/runs/29210780582)
  (all 10 jobs green) and release run [29210860451](https://github.com/philpaz/recusal/actions/runs/29210860451)
  (full suite at the release commit, hash-locked install, `--no-isolation` build,
  neutral-directory wheel check, Trusted Publishing).
- **Acknowledging removal of every pinned server refuses precisely** (verbatim):

```text
FAILED mcp_full_decommission_unsupported [CRITICAL]: every pinned server is acknowledged as removed and nothing was observed; an empty observation certifies nothing, and the manifest keeps authorizing all pinned runtime names regardless ...
```

- **An unhashable sequence member raises the documented exception** (verbatim):

```text
ValueError: observation unverifiable must be unique nonempty server names
```

The same properties are pinned as deterministic tests in
[`tests/test_mcp_observation_decommission.py`](../tests/test_mcp_observation_decommission.py).

## Release proofs on the record (v0.5.6, 2026-07-12)

For v0.5.6 (commit `c5dd3d5`):

- **Workflow evidence is public**: CI run [29209930133](https://github.com/philpaz/recusal/actions/runs/29209930133)
  (all 10 jobs green) and release run [29209981300](https://github.com/philpaz/recusal/actions/runs/29209981300)
  (full suite at the release commit, hash-locked install, `--no-isolation` build,
  neutral-directory wheel check, Trusted Publishing).
- **A wholly omitted pinned server refuses; a deliberate removal is explicit**
  (verbatim: two servers pinned, a one-server dump verified):

```text
FAIL - 1 CRITICAL failure(s) - refused, no retry.
  FAILED mcp_server_unobserved [CRITICAL]: pinned server 'banking' is absent from every component of this observation and was not explicitly marked removed; ...
```

  and with `--removed banking` the same verify exits 0, recording:

```text
  warning mcp_server_removed: pinned server 'banking' is acknowledged as deliberately removed (recorded, not blocking); re-pin so the manifest stops authorizing its runtime names
```

- **A reserved Claude server name refuses before its command can run** (a config
  entry named `workspace` whose command would write a marker file):

```text
FAIL (failed closed) - could not read the configuration: ... this name is reserved by Claude Code for a built-in server ...
```

  The marker file does not exist afterward; the configured command never executed.

The same properties are pinned as deterministic tests in
[`tests/test_mcp_observation_inventory.py`](../tests/test_mcp_observation_inventory.py),
including the counter-instrumented proof that an unpinned MCP call never invokes the
wrapped business policy.

## Release proofs on the record (v0.5.5, 2026-07-12)

For v0.5.5 (commit `b1bcddf`):

- **Workflow evidence is public**: CI run [29208747925](https://github.com/philpaz/recusal/actions/runs/29208747925)
  (all 10 jobs green) and release run [29208815312](https://github.com/philpaz/recusal/actions/runs/29208815312)
  (full suite at the release commit, hash-locked install, `--no-isolation` build,
  neutral-directory wheel check, Trusted Publishing).
- **Source omission cannot bypass a pinned launch identity** (verbatim library proof:
  matching catalog plus matching instructions, `sources` omitted, against a pin
  carrying a stdio source):

```text
failed checks: ['mcp_source_unobserved']
PROOF OK: matching catalog + matching instructions, omitted sources -> CRITICAL refusal
```

- **A malformed event in a reused process carries no stale digest** (verbatim: one
  process, one policy object, a valid MCP event then malformed JSON through the real
  hook with `audit=`):

```text
valid event control:     {"manifest_sha256": "sha256:a854a5c8...", "recusal_version": "0.5.5"}
malformed event control: {"recusal_version": "0.5.5"}
```

- **WebSocket OAuth refuses through the real CLI** (a `ws` entry carrying `oauth` in
  `.mcp.json`; nothing is written):

```text
FAIL (failed closed) - could not read the configuration: ... Claude Code documents
WebSocket MCP authentication as header-only (HTTP supports OAuth, WebSocket does not) ...
```

The same properties are pinned as deterministic tests in
[`tests/test_mcp_observation_strict.py`](../tests/test_mcp_observation_strict.py).

## Release proofs on the record (v0.5.4, 2026-07-12)

Every release claim above "it passed" is anchored to reproducible evidence. For v0.5.4
(commit `7c456a3`):

- **Workflow evidence is public**: CI run [29207262548](https://github.com/philpaz/recusal/actions/runs/29207262548)
  (all 10 jobs green, including the first exercise of the hash-locked
  release-environment job) and release run [29207317542](https://github.com/philpaz/recusal/actions/runs/29207317542)
  (full suite at the release commit, `--require-hashes` install, `--no-isolation`
  build, neutral-directory wheel check, Trusted Publishing).
- **The instructions rug pull, live through the CLI** (verbatim, tools byte-identical,
  only the initialize-result `instructions` rewritten, single-server `--server` path):

```text
  [ok] mcp_launch_spec: server 'banking' launch specification matches the pin
  [ok] mcp_manifest: 1 tool declaration(s) across 1 server(s) match the pinned manifest
FAIL - 1 CRITICAL failure(s) - refused, no retry.
  FAILED mcp_instructions_changed [CRITICAL]: server 'banking' changed its instructions (the discovery-influence rug pull); refusing until a human re-reviews and re-pins
```

- **Audit provenance, live through the real hook** (a `manifest_policy` gate with
  `audit=`, the caller attempting to forge the manifest digest via `control=`): the
  pinned MCP call's audit entry records the digest of the manifest bytes the
  invocation actually verified, the forged value is stripped, and the following
  non-MCP call's entry carries no manifest digest at all:

```text
mcp call control:     {"manifest_sha256": "sha256:e2354ab7...", "policy_id": "proof-054", "recusal_version": "0.5.4"}
non-mcp call control: {"policy_id": "proof-054", "recusal_version": "0.5.4"}
```

Reproduce both from any checkout: the rug pull with `recusal mcp pin/verify --from
<dump> --server <name>` (change only the dump's `instructions` between the two), the
provenance proof by piping two PreToolUse payloads into a gate built exactly as in
[cookbook recipe 16](COOKBOOK.md). The same properties are pinned as deterministic
tests in [`tests/test_mcp_observation.py`](../tests/test_mcp_observation.py), so this
page cannot silently rot.

## Honest scope

This proves the **enforcement path** end to end on the real wire format: a real hook, the
real PreToolUse JSON, a real `deny` that Claude Code honors. This repository does not claim fleet-scale
deployment evidence (tamper-evident audit logging ships in `recusal.audit`). What it does claim is true: Recusal
refuses real, dangerous tool calls in the tool people actually use, and it does so to its
own maintainers first.
