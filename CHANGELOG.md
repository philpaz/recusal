# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.4.2] - 2026-07-12

The audit-integration release plus a hardening pass driven by a third external review,
which found (and live Windows validation confirmed) that the POSIX launcher fails OPEN
under Claude Code's documented PowerShell fallback, and that the shared-file audit
pattern was unsafe under Claude Code's documented parallel hook execution.

### Fixed
- **The gate no longer fails open on Windows without Git Bash (P0).** Claude Code runs
  shell-form hooks under Git Bash and falls back to PowerShell when it is absent, where
  the POSIX launcher is a parse error with exit 1 - a NON-blocking code, i.e. the gate
  silently disabled (live-verified). `recusal init` is now platform-aware: on Windows it
  registers a PowerShell-native launcher with an explicit `"shell": "powershell"`
  (PowerShell is always present; the explicit shell also keeps Git Bash from trying to
  parse it), POSIX elsewhere, `--launcher both` for a settings.json shared across OSes.
  The PowerShell launcher was validated end to end on a real Windows host: deny emits
  the JSON and exits 0, a broken gate and a missing interpreter each coerce to exit 2.
  `recusal doctor` now validates the registered launcher's shell strategy against the
  host instead of grepping for "exit 2"; the plugin's POSIX launcher is scoped honestly
  (macOS/Linux/Windows-with-Git-Bash) in its own manifest.
- **File-backed audit appends are one serialized transaction.** Claude Code runs hooks
  for parallel tool calls concurrently; two hook processes could read the same head and
  write sibling entries, forking the chain with neither append reporting an error. An
  append now holds an inter-process lock (`<path>.lock`), re-reads the chain head from
  the END of the file, writes, and only then commits in-memory state - so a failed write
  never advances the chain (previously state advanced before persistence). `fsync=True`
  opts into durability past the OS cache. Proven by a test that hammers one file from
  four real processes and verifies one gapless chain.
- **`resume="tail"` now actually recovers the head from the final record** (backward
  seek, corrupt-tail tolerant, full-scan fallback for pathological logs) instead of
  streaming the whole file - making it O(final record) in time as well as memory. The
  README's "flat per-call cost" claim was wrong for the old implementation and was
  corrected along with the code.
- **`verify` and `verify_file` return verdicts on garbage, never crash out of one.** A
  valid-JSON line that is not an object (`[]`, `null`, a string, a number) is a named
  verification failure instead of an uncaught exception; hash/seq/decision shapes are
  validated; unreadable files (permissions, a directory, invalid UTF-8) return a
  structured failure; and the CLI now routes through the same strict verifier instead of
  duplicating the logic (unreadable stays exit 2, broken chain stays exit 1).
- **The manifest cache cannot serve stale authorization.** The (mtime, size) signature
  missed a same-size, timestamp-preserved replacement - exactly how a REVOCATION might
  land via deployment tooling - and had a stat-then-open race. `manifest_policy` now
  reads the manifest bytes every call and reparses only when their SHA-256 changes;
  pinned with a test that revokes a tool under a preserved mtime and identical size.
- **The stdio observer bounds aggregate hostile input.** Bounded reader queue, a total
  character budget (`MAX_TOTAL_CHARS`) and an unrelated-message budget
  (`MAX_UNRELATED_MESSAGES`) across the observation; a non-object entry in `tools`
  refuses the whole catalog (silently filtering would certify a subset as the declared
  surface); JSON nested beyond parseable depth on the wire is an `McpFetchError`, not a
  RecursionError; `NaN`/`Infinity` are rejected as unparseable; `nextCursor` must be a
  bounded string.
- **The GitHub Action ref now selects the implementation unconditionally.** The install
  step force-installs the action ref's bundled source, replacing whatever is on the
  runner; `use-installed: "true"` is the explicit escape hatch for a deliberately
  preinstalled checkout (the dogfood job names it now). CI gained a provenance job that
  proves both the clean-runner install and that a deliberately conflicting preinstalled
  recusal gets replaced.
- **A null/empty/non-string `tool_name` is a malformed envelope**, failing closed before
  any policy is asked to reason about it.

### Added
- **`run_pretooluse_hook(audit=...)`: every adjudication on the record, one wire.**
  Pass an `AuditLog` and every hook decision - defer, allow, and deny alike - appends one
  hash-chained entry naming the tool, the decision, the reasons, and a SHA-256 fingerprint
  of the proposed `tool_input` (contents are never embedded: a `Write`'s file body or an
  env value must not leak into the log). `actor=` labels entries and defaults to the
  event's `session_id`. An unwritable log fails **closed** to a deny - the record is part
  of the control - unless `fail_closed=False`. Malformed-event and policy-error denials
  are on the record too, with synthesized findings saying what happened.
- **`AuditLog(path, resume="tail")`: resume the chain without holding the log in memory.**
  Recovers the chain head from the final record and retains no entries, before or after;
  appends go to disk only. The default `resume="full"` is unchanged. Tail is the right
  mode for a per-call hook over a growing log and for long-running gates; verify with
  `verify_file(path)`.
- Audit records carry `prompt_id` (transcript linkage, from the documented PreToolUse
  event; a `tool_use_id` is recorded defensively should the event ever include one), and
  `verify_file` takes the same `expected_head=` anchor as `verify`.

### Documentation
- Claims squared with implementation: "same evidence, same verdict, forever" is now
  "same evidence, same policy, same version, same verdict" (FAQ, CONSTITUTION); the
  Windows launcher scope is stated everywhere the launcher appears; prompt-time `@file`
  references are named as outside PreToolUse (SECURITY.md); and the audit docs state
  plainly that finding messages are plaintext record content - keep secrets out of
  messages.

## [0.4.1] - 2026-07-12

Hardening and documentation only, driven by two external reviews; no new capabilities.

### Added
- **`fetch_tools_stdio(minimal_env=True)` and `recusal mcp pin/verify --minimal-env`.**
  By default the spawned server inherits the full parent environment (matching how Claude
  Code launches the same server). A server being *pinned* is by definition not yet
  trusted, so `minimal_env=True` hands it only what a process needs to launch (PATH and
  friends) plus explicitly passed `env` - an API key in your shell does not ride along.

### Fixed
- **`verify_file` is a strict verifier: a malformed nonblank line is a failure, not a
  skip.** `load()` tolerantly skips a corrupt line (a half-written tail must not brick a
  *reader*), and `verify_file` verified only what `load()` kept, so a log whose newest
  records were garbage could read as intact; the CLI already refused this, the library
  helper now matches it. A missing file is a failure too (a missing log is not an intact
  log), and `verify_file` now takes the same `expected_head=` anchor as `verify`.
- **The GitHub Action's ref now selects the implementation that runs.** The install step
  defaulted to "latest from PyPI" when recusal was not already importable, so
  `uses: philpaz/recusal@vX` could execute a *later* release than the pinned action. The
  default now installs the package bundled with the selected action ref
  (`$GITHUB_ACTION_PATH`); an explicit `version:` input remains as the one deliberate
  override, and a job that pre-installed a checkout keeps it.
- **The stdio fetcher treats `initialize` negotiation as binding.** The response's
  `protocolVersion` must be one this client speaks (`SUPPORTED_PROTOCOL_VERSIONS`:
  2025-11-25 through 2024-11-05; the newest is now requested), the server must return a
  capabilities object, and it must advertise the `tools` capability - each failure is a
  refusal, not a shrug-and-proceed.
- **The declaration screen returns a verdict on hostile nesting instead of crashing.**
  A ~3000-deep `inputSchema` blew the recursion limit inside `screen_tool_declarations`,
  so `recusal mcp pin` died with a RecursionError traceback - fail-closed by accident,
  but a crash is not a verdict. The walk is iterative now, and nesting past
  `MAX_DECLARED_DEPTH` (200) is itself an ERROR finding (`mcp_declaration_depth`): too
  deep to plausibly review gets the same treatment as too long. Depth beyond what
  canonical JSON can serialize now fails closed (exit 2) in `pin`/`verify` as well.
- **The stdio reader bounds a single line (`MAX_LINE_CHARS`).** A server emitting one
  endless line with no newline buffered unboundedly until the timeout; it now refuses
  with a truthful "runaway stream" error.
- **String `passed` values now fail closed on unrecognized tokens, matching `status`.**
  `Finding.coerce` read a string `passed` against a false-token blocklist, so an
  unrecognized token (`"passed": "maybe"`) coerced to PASS while `"status": "maybe"`
  failed closed. Both fields now share the allowlist posture: a string counts as a pass
  only when it is an affirmative token (`"true"`/`"yes"`/`"1"`/`"pass"`/...); anything
  unrecognized reads as a failure. Genuine booleans and numbers are unchanged.

### Changed
- **`manifest_policy` caches the pinned names, keyed by the manifest file's
  (mtime, size).** An unchanged manifest costs a `stat` plus a set lookup per call
  instead of a read+parse+validate; a re-pin is picked up live, a deleted or corrupted
  manifest still fails closed (the stale pin is never served past its file).
- **The MCP control plane is protected by default.** `.mcp.json` decides which server
  processes launch and `mcp-manifest.json` is what "approved" means at call time, so both
  join the deny-list's default protected paths (kill-switch rank); cookbook recipe 13 now
  composes `manifest_policy` over `deny_list_policy()` so the pin protects its own files.
- **Releases prove the release commit.** `release.yml` runs the full gate (ruff, format,
  mypy, pytest) at the exact release commit before anything builds or publishes, and every
  third-party action in CI and release workflows is pinned to an immutable commit SHA (a
  moving tag is a rug-pull surface). mypy now type-checks against Python 3.9, the declared
  minimum, instead of 3.10.

### Documentation
- **The launch-identity boundary is named everywhere it matters.** Observing a stdio
  catalog *executes the command the config declares*, and the manifest pins the declared
  catalog, not the identity of the process that declares it - a rewritten `.mcp.json`
  runs at observe time, before its catalog can fail verification. README, SECURITY.md,
  the cookbook, the module docstrings, and the `--stdio`/`--claude-config` CLI help all
  now say so plainly ("treat the config as executable code"); pinning launch
  specifications (manifest v2) is a named roadmap item, not shipped.
- **Claims tightened to what the implementation proves.** "The agent could not subvert
  it" is restated as: within a correctly registered routed tool channel, an unapproved
  capability is refused by default rather than inferred safe. "Closes the discovery
  boundary" becomes "adds deterministic integrity controls at the discovery boundary".
  Cookbook recipe 15 is "the three-boundary MCP governance pattern", not "the full MCP
  governance stack". "Independent" is defined once in the README: the verdict is produced
  outside the model's decision path; deployment isolation remains the adopter's
  responsibility. And read-only is stated as *nonmutating*, not confidentiality-safe: the
  default-safe tools can still read credentials, so add path/subject-level read rules
  where confidentiality matters.
- The `_SHELL_META` comment in the Claude Code allowlist now states the actual posture:
  glob (`*`, `?`, `[`) and tilde expansion are accepted for allowlisted read-only
  binaries (any literal path is equally readable by design, and expansion can never
  select the binary itself, since argv[0] must literally match the allowlist); the
  metacharacter set refuses chaining, substitution, redirection, and escapes. Pinned
  with tests in both directions.

## [0.4.0] - 2026-07-10

### Added
- **MCP discovery governance (`recusal.mcp`): pin the tool catalog, refuse the rug pull.**
  The model chooses tools by reading their declared descriptions, so the discovery
  boundary (`tools/list`) is where a poisoned description or a post-approval definition
  change steers the agent before any call exists for a call-time policy to see. This
  release closes that boundary the way the library closes every boundary: deterministic
  evidence through the same kernel, with the human where the judgment is.
  - **The kernel**: `build_manifest` pins a reviewed catalog to a deterministic manifest
    (SHA-256 fingerprints over canonical JSON, byte-exact, no unicode normalization, so a
    homoglyph swap IS a change; hashes only, a poisoned description is never embedded;
    no timestamp inside, *when* belongs to the audit log). `diff_manifest` emits Findings:
    an unpinned server or tool and a changed declaration (the changed fields are named;
    a changed *description* is called out as the rug-pull vector) are CRITICAL; a removed
    tool or absent server is a recorded WARNING; an empty or ambiguous observation fails
    closed. `screen_tool_declarations` is a pin-time review aid (deterministic injection
    markers + a size cap over the *whole* declaration — title, annotations, schema property
    names/descriptions, enum values — ERROR → RETRY → a human looks), deliberately not a
    malice detector: whether a declaration is malicious is semantic judgment, made by the
    human at pin time; everything after the pin detects *change*, not *intent*.
  - **CLI**: `recusal mcp pin` / `recusal mcp verify`, same exit-code discipline as the
    other CI commands (0 clean, 1 needs-review, 2 refused/drift/operational error). The
    pin fails toward refusal three ways: an incomplete observation refuses, a flagged
    description screen refuses to write until `--force` records human review, and
    replacing a differing manifest refuses without `--update`. Sources: a live stdio
    server (`--stdio NAME COMMAND`), every stdio server in a Claude Code `.mcp.json`
    (`--claude-config`, URL-based servers are surfaced as unfetchable, never silently
    dropped), or a JSON dump (`--from`, the escape hatch for HTTP servers).
  - **Call-time enforcement**: `manifest_policy("mcp-manifest.json")` drops into the same
    `PreToolUse` gate and refuses any `mcp__server__tool` call that was never pinned
    ("no pin, no MCP"); a missing or corrupt manifest fails CLOSED for MCP calls; wraps
    an inner policy so argument-level rules compose on top of the pin.
  - **The fetcher** (`recusal.mcp_fetch`, a separate module — the one place in the package
    that spawns a process, kept apart so the decision surface stays pure/stdlib): a minimal
    zero-dependency stdio MCP client (`fetch_tools_stdio`, newline-delimited JSON-RPC,
    `initialize` → `notifications/initialized` → paginated `tools/list`). Collection is
    never decision; every irregularity (timeout, early exit, JSON-RPC error, unparseable
    line, invalid UTF-8) raises so a failed observation can never read as an empty,
    clean-looking catalog.
  - **Proof**: `examples/mcp_manifest_rugpull.py` (offline demo: pin → rug pull → FAIL →
    unpinned call refused) and 73 new tests across `tests/test_mcp_manifest.py`,
    `test_mcp_policy_bridge.py`, `test_mcp_fetch.py` (a real fake-server subprocess:
    pagination, notifications, stderr noise, timeout, early exit, invalid UTF-8), and
    `test_mcp_cli.py`.
  - **What this does and does not do (honest scope).** It governs *discovery-time* and
    *call-time*: `verify` proves the catalog at the moment it runs (wire it into CI and
    session start), and `manifest_policy` enforces approved-tools-only on each call by
    name. It is **not** a live tap on every message: a server that serves a clean catalog
    to `verify` and a poisoned one to the live session (a client- or time-discriminating
    server) is a residual this layer names, not one it closes. The description screen is a
    **deny-list** review aid with a deny-list's ceiling (known injection phrasing across
    the whole declaration, not just `description`), not a malice detector. MCP's flat
    `mcp__server__tool` runtime naming means a pinned tool whose name contains `__` shares
    a runtime string with another split; pinning one authorizes either (inherent, not
    removable at this layer). Transport/authorization threats (confused deputy, token
    passthrough, session hijacking) remain the MCP spec's own Security Best Practices layer.
  - **Pre-release hardening** (two independent audits, correctness + adversarial): `verify`
    fails closed (not an uncaught traceback) on an uncanonicalizable string; the manifest
    validator rejects a non-object `fields` entry before `diff` dereferences it; a pinned
    server silently swapped to a URL transport is a CRITICAL refusal (not a WARNING that
    passes); `--json` output is always valid JSON on every branch (notes and prose no
    longer interleave with the payload); a server observed with zero tools is a shrunk set
    (WARNING), consistent with `build_manifest`, not conflated with a failed fetch; a
    `--from` mapping whose server is literally named `tools` no longer drops its siblings
    (mode is chosen by `--server`); invalid UTF-8 from a server surfaces a truthful error
    instead of "server exited"; the manifest write is atomic; the fetch caps tool count;
    and a killed child is reaped.
- **MCP tool governance at the call boundary, documented and pinned.** MCP server tools reach Claude Code's
  `PreToolUse` hook as ordinary tools named `mcp__<server>__<tool>`, so the existing
  `policy(tool_name, tool_input)` seam and the `.*` matcher already govern MCP calls with
  no MCP-specific adapter; this makes that capability explicit instead of implied.
  - README section **"MCP tools, the same gate"**, including the three-boundary model
    (discovery / invocation / response) with today's coverage stated honestly: invocation
    is this gate, response is the injection-quarantine recipe, discovery (tool-description
    poisoning, manifest/schema drift) is named as a boundary Recusal does not collect
    evidence for yet.
  - `examples/mcp_governance.py`: runnable demo and real hook (`--hook`) with
    approved-server pinning, destructive-verb refusal, repo scope, write-path confinement,
    and allowlist mode (an MCP tool is refused unless affirmatively named).
  - `tests/test_mcp_governance.py`: pins the claims, deny/defer at the `decide` seam, a
    real `PreToolUse` event carrying an `mcp__` name end to end, allowlist default-deny
    for MCP, and fail-closed on a buggy policy adjudicating an MCP event.
  - Cookbook recipe 12 (**Govern MCP tool calls**) and two verified sources in
    `docs/REFERENCES.md` (Claude Code hooks reference; MCP spec *Security Best Practices*,
    noting its authorization/transport scope is complementary, not overlapping).

## [0.3.0] - 2026-07-08

### Added
- **CI adjudication commands.** The `recusal` CLI grew three subcommands that expose the
  kernel to CI with blocking exit codes (`PASS` → 0, `RETRY` → 1, `FAIL` → 2; every
  operational error — unreadable file, invalid JSON, malformed anchor — exits 2,
  indistinguishable from FAIL on purpose: a gate that cannot adjudicate must refuse, not
  wave the job through). All three take `--json` for a stable machine-readable payload.
  - **`recusal verdict findings.json`**: adjudicate any tool's findings file (a JSON array,
    or an object with a `findings` array; `-` reads stdin). Strict by default — a finding
    that omits `status`/`passed` is rejected rather than read as a silent pass
    (`--lenient` opts out) — and an *empty* findings set fails closed: an evidence set
    that proves nothing certifies nothing (the `GateAdjudicator` rule, at the CLI seam).
  - **`recusal audit verify log.jsonl [--expect-head COUNT:HASH]`**: verify a hash-chained
    audit log. A **missing log fails closed** (a missing log is not an intact log), and a
    nonblank line that does not parse counts as a *break*: `recusal.audit.load` skips such
    a line so a reader survives a half-written tail, but a verifier that ignored it could
    bless a log whose most recent entries are unreadable.
  - **`recusal doctor [--dir]`**: health-check a scaffolded gate — gate script present and
    compiling, hook actually registered in `settings.json`, launcher coercing failures to
    the blocking exit code — so "the gate silently isn't installed" is caught by CI
    instead of discovered during an incident. The doctor is adjudicated by the same kernel
    it checks: its observations are `Finding`s folded through `compute_verdict`.
- **GitHub Action** (`action.yml`): the same three commands as a composite action
  (`uses: philpaz/recusal@v0.3.0`), so a refusal blocks a merge, not just a tool call.
  Inputs flow through `env`, never interpolated into the shell body (no injection seam);
  given nothing to adjudicate it exits 2 rather than pass vacuously. Dogfooded by this
  repository's own CI, including the negative case: a tampered audit log must make the
  gate refuse (`tests/test_cli.py` drift-locks all of these properties).
- `recusal --version`.

## [0.2.0] - 2026-07-07

### Added
- **`python -m recusal init`** (also installed as the `recusal` console script): one-command
  scaffolding of the Claude Code gate. Writes `.claude/hooks/recusal_gate.py` (a thin shim
  over the shipped `deny_list_policy()`; `--posture allowlist` emits the default-deny
  variant with `--writable-root`) and registers the fail-closed interpreter-probing
  launcher in `.claude/settings.json`. Fail-safe by construction, pinned by tests: an
  existing gate file is never overwritten, an existing `settings.json` is merged (never
  clobbered) and left byte-for-byte untouched if it does not parse, and re-running is a
  no-op. A drift-lock test asserts the emitted launcher stays byte-identical to the one
  this repository registers for itself in `.claude/settings.json.example`.
- **Claude Code plugin (`recusal-gate`)**: the repo is now a plugin marketplace
  (`.claude-plugin/marketplace.json` + `claude-plugin/`), so the gate installs user-wide with
  `claude plugin marketplace add philpaz/recusal && claude plugin install recusal-gate@recusal`.
  The plugin wires the same deny-list shim through the same fail-closed launcher (drift-locked
  to the canonical command by `tests/test_claude_plugin.py`); if the `recusal` package is not
  pip-installed it refuses every tool call rather than silently disabling itself. Verified
  live: a marketplace-installed plugin refused `rm -rf` in a session running under
  `--dangerously-skip-permissions`.
- **README demo GIF** rendered from two verbatim transcripts: a live Claude Code session in
  which the dogfooded hook refuses `rm -rf` under `--dangerously-skip-permissions`, and the
  offline `examples/claude_refusal.py` run.

## [0.1.3] - 2026-07-06

### Added
- **`recusal.deny_list`**: the reference deny-list engine, extracted from the dogfood hook
  into an importable, versioned, unit-tested module. `deny_list_policy(...)` builds the
  hardened policy (destructive shell, secret writes, kill-switch edits/deletes, with
  de-obfuscation, pipe-into-any-interpreter, reverse-shell, `cd`/variable-indirection, and
  best-effort symlink coverage) and takes `protected_paths=` / `secret_basenames=` /
  `command_keys=` / `read_only_tools=` so adopters can point it at their own gate;
  `analyze_command(...)` exposes the command adjudicator as a pure function.

### Changed
- `.claude/hooks/recusal_gate.py` is now a thin shim over `recusal.deny_list.deny_list_policy()`,
  so a fix to the deny-list ships through `pip install -U` instead of a copy-paste, and the
  security logic is covered as a package unit (`tests/test_deny_list.py`) in addition to the
  end-to-end dogfood test that still loads the real hook.

## [0.1.2] - 2026-07-06

### Security
- **Fail-closed on a `status` failure token (HIGH).** `Finding.coerce` read the `status`
  field against a hardcoded `{"fail","error","warn"}` blocklist, so a CRITICAL finding with
  `status` of `"failed"`, `"false"`, `"0"`, `"no"`, `"denied"`, `"fatal"`, … coerced to
  **PASS**, even under `strict=True`, and passed through every enforcement adapter as
  allow/defer. This was the exact `bool("false")==True` silent-pass this library exists to
  prevent, on the `status` path. The field is now read against a pass **allowlist**: any
  token not affirmatively passing (`pass`/`passed`/`ok`/…) fails closed.
- **Empty gate evidence no longer passes vacuously (MED).** `GateAdjudicator.adjudicate`
  with an empty findings set returned a PASS, letting a gate that proved nothing count
  toward `release_ready`. It is now a CRITICAL `evidence_error`, matching the module's
  "absence of evidence is not a pass" contract.
- **Dogfood hook: closed a self-protection bypass and enumeration gaps.** `cd .claude && rm
  settings.json` (and variable-indirected `d=.claude; rm $d/settings.json`) split the
  protected path across the `&&`, so the contiguous-substring self-protect never matched;
  a `cd`/`pushd` into, or variable binding of, a control dir plus a write verb is now
  refused. Added `php -r` / `lua -e` / `Rscript -e` / `groovy`/`elixir` inline-exec forms to
  the kill-switch write guard, and caught the spaced form of the classic fork bomb.

### Fixed
- Audit-log claims corrected across README, SECURITY, HOWTO, WHY, CHANGELOG, and the module
  docstrings: an in-place edit is caught only for an entry that still has an untampered
  successor; a tail-suffix rewrite (down to the last entry) and a forged append pass
  unanchored `verify` and need the `expected_head` anchor. Pinned by a new regression test.
- Purged em/en-dashes from all shipped files (regressed in the 0.1.1 copy pass).

## [0.1.1] - 2026-07-06

### Security
- **Allowlist default hardened.** Removed `pytest`, `mypy`, `rg` (and `ruff`) from
  `DEFAULT_SAFE_BINARIES`: each executes arbitrary code through an argument (pytest imports
  `conftest.py`, mypy loads plugins, rg `--pre` spawns a command), which reopened the
  write-a-script-then-run-it bypass allowlist mode exists to close. The default set is now
  read/inspect tools only; any binary you add must be safe under every argument. COOKBOOK
  recipe 11 and `examples/allowlist_gate.py` updated to match.
- **Dogfood hook self-protection widened.** Closed fail-open gaps where inline-interpreter
  code (`py -c`, `python3.12 -c`, `node --eval`, `deno`/`bun eval`) or a slash-decorated
  control-directory move (`mv ./recusal`, `mv .claude/`) could edit/move the gate's own
  package or config and defer. `writable_root` now resolves symlinks (no escape via an
  in-root link). `sed -n` reads no longer false-positive (only `sed -i` writes are refused).
- **ReDoS fixed** in path normalization (trailing-dot stripping is now linear, not
  quadratic); the `[IO.File]::Write*` pattern quantifier is bounded.

### Changed
- Docs frame deny-list vs allowlist as two paths chosen by channel (no ranking); the
  reference architecture dogfoods the deny-list path, with the rationale stated.

## [0.1.0] - 2026-07-02

### Added
- **Evidence contract**: `Finding`, `Verdict`, `Severity`, `Decision`, and
  `compute_verdict` (the typed, zero-dependency spine). See `docs/EVIDENCE.md`.
- **Built-in checks**: `row_count`, `null_rate`, `referential_integrity`, `in_set`,
  `in_range`, `required_keys` (operate on any dict-like rows; no pandas required).
- **`GateAdjudicator`**: staged `G0`-`G8` release checkpoints and a release-evidence rollup.
  Each gate is `compute_verdict` applied at a checkpoint, returning a typed `GateResult`
  (wrapping a `Verdict`); `release(...)` rolls them into a `ReleaseEvidence`. Domain-neutral,
  gates are pure `(id, description)` labels you can replace, and the rollup is a pure
  function of the findings (no timestamps, no nondeterminism), so it replays and compares
  exactly. One decision function across the whole library.
- **Claude adapters**: `recusal.claude` (manual-loop tool gate + Managed Agents
  confirmation) and `recusal.claude_code` (a `PreToolUse` hook that denies even under
  `bypassPermissions` and fails closed on a policy error).
- **Allowlist mode as library API**: `recusal.claude_code.allowlist_policy`, a
  default-deny policy factory (nothing runs unless affirmatively named: vetted first
  binaries only, no shell metacharacters, **bare interpreters refused** so
  `python script.py` cannot execute unvetted code, writes scoped to a `writable_root`,
  per-tool `allow` predicates). Closes the write-a-script-then-run-it bypass a deny-list
  cannot see; pinned in `tests/test_claude_code_allowlist.py`. The docs now carry
  **two-tier claim language**: a deny-list "raises the cost / stops the common cases";
  only allowlist mode earns "the agent could not subvert it," scoped to the routed tool
  channel (HOWTO §1, README, SECURITY, FAQ).
- **Tamper-evident audit log**: `recusal.audit` (`AuditLog`, `verify`): a hash-chained,
  append-only JSONL record of every verdict; an edit or reorder of any entry with a
  surviving successor is detected, and tail truncation, tail-suffix rewrite, or a forged
  append are caught with the `expected_head` anchor. Maps to OWASP Agentic logging / EU AI
  Act Article 12 (record-keeping).
- **Deterministic failure classifier**: `recusal.classify` (`classify_failure`,
  `classify_verdict`): routes a failure to a class + remediation channel (policy_violation,
  prompt_injection, transient, code_bug, data_shape, data_missing, spec_ambiguity)
  by explicit markers; extensible taxonomy, never guesses.
- **Dogfood**: Recusal governs its own repository via a real Claude Code hook; verbatim,
  reproducible, CI-locked proof in `docs/PROVEN.md`.
- **Examples**: offline refusal demo, live Claude-agent demo, an OWASP-mapped scenario
  gallery, a Claude Code hook, an audit-log demo, and a **framework-neutral agent loop**
  (`examples/agent_loop.py`) whose only import is `recusal`, proof the zero-dep core gates
  any loop with no Claude and no SDK.
- **Docs**: `CONSTITUTION`, `docs/WHY`, `docs/EVIDENCE`, `docs/HOWTO`, `docs/EXTENDING`,
  `docs/LANDSCAPE`, `docs/PROVEN`, a `docs/FAQ` (adoption objections answered), a
  `docs/COOKBOOK` (copy-paste policies for the common gated actions), and a `docs/README`
  documentation index.
- **Community files**: `CODE_OF_CONDUCT` (Contributor Covenant), GitHub issue templates
  (bug / feature) and PR template under `.github/`, and enriched package metadata
  (`project.urls`, `Typing :: Typed`).
- **Tooling**: zero runtime dependencies; ruff (lint + format), mypy, pytest, and
  pre-commit, all run in CI; a `release.yml` workflow that builds and publishes to PyPI via
  Trusted Publishing (OIDC) on a GitHub Release.

### Security & hardening
- The Claude Code hook **fails closed on a malformed/non-object event**, not just a policy
  exception (previously a garbled event deferred, i.e. failed open).
- `compute_verdict(..., strict=True)` / `Finding.coerce(..., strict=True)` reject a loose
  evidence dict that omits an explicit `status`/`passed` instead of treating it as a pass.
- `recusal.audit`: precise tamper model (tamper-evident, not tamper-proof);
  `verify(..., expected_head=(count, last_hash))` catches truncation and a full-chain rewrite;
  resume tolerates a corrupt trailing line; `default=str` so a verdict is never dropped.
- `recusal.classify`: tightened over-broad default markers (no longer mis-escalates benign
  validation errors to `refuse`, or numeric substrings to `retry`); non-string input is
  coerced; `classify_verdict` returns `pass -> proceed` on a PASS.
- `GateAdjudicator`: a release is not "ready" with empty or missing gate evidence.
- The dogfood hook protects its own settings/hook from being disabled, normalizes commands,
  and matches `rm` recursive-force in any flag order; example/cookbook path checks use
  `os.path.commonpath` (the `startswith` prefix bypass is fixed).
- **Dogfood hook, red-team hardening pass** (closes bypasses found in a full adversarial
  review, each pinned as a regression test in `tests/test_dogfood_redteam.py`):
  self-protection now covers *removal* of the kill-switch (`rm`/`mv`/`del` of the hook or
  settings), not just edits; secret and self-protect checks run against the de-obfuscated
  command and three path readings, so a quote-split (`.cla""ude`), a backslash-escape
  (`.cl\aude`), or a Windows separator can no longer hide a protected path; piping into
  *any* interpreter (`python`/`perl`/`ruby`/`node`/`php`/`pwsh`), not just `sh`/`bash`, is
  refused; recursive `rm` is refused even without `-f`; `git clean -f`, `git checkout --`,
  `find -exec rm`, `unlink`, and reverse/bind shells (`/dev/tcp`, `nc -e`) are refused;
  `prod.env`-style secret files are protected. A false-positive guard test keeps reads
  (`cat`/`grep`) and running the hook (`python file.py`) deferring, so the gate stays usable.
  Coverage is no longer Bash-only: any tool carrying a command under `command`/`cmd`/
  `shell`/`script` (an MCP shell) gets the same analysis, and a generic kill-switch guard
  refuses any non-read tool (an MCP filesystem tool) that targets a protected control path;
  `git config core.hooksPath` and `.git/hooks/**` writes (code-exec-on-commit vectors) are
  refused. Remaining limits are documented explicitly in `SECURITY.md`: network egress
  (an allowlist recipe, not the baseline hook), symlink/TOCTOU (closed for tool-based writes
  by the realpath layer below; `Bash` fragments stay string-matched), and runtime-constructed
  command names (the deny-list ceiling).
- **Subversion test library + second red-team pass** (`tests/test_subversion_*.py`, ~170
  adversarial cases across kernel, adapters, audit, hook, and classifier). New fixes it drove:
  `Finding.coerce` reads a stringified `"passed": "false"`/`"no"`/`"0"` as a *failure* instead
  of trusting raw truthiness (a `bool("false")` is `True`), closing a silent-pass footgun at
  the loose-dict boundary; the dogfood hook now matches command-carrying keys
  **case-insensitively and at any nesting depth** and joins **argv-array** command values
  (so `"Command"`, `{"payload": {"command": ...}}`, and `["rm","-rf","/"]` can't smuggle a
  shell past); and `socat EXEC:`/`SYSTEM:` reverse shells are refused. The suite also *pins*
  the honest deny-list limits as expected-defer tests (runtime-constructed names, interpreter
  code) so the boundary is a tested fact, not a footnote.
- **Best-effort `realpath` layer for tool-based writes** (closes the innocent-name TOCTOU):
  a `Write`/`Edit` or MCP filesystem write whose path resolves through a symlink onto a
  protected control path is refused (`_resolves_into_protected`), so `notes.txt` ->
  `.claude/settings.json` is caught even though the path string carries no protected segment.
  A not-yet-created link and `Bash` string fragments remain out of scope by design (an
  allowlist is the real defense).
- **Symlink resolution now covers a bare filename on the MCP path** (found by running the
  subversion suite in an environment that grants symlink privilege, where the case was live
  rather than skipped): the generic kill-switch guard used to symlink-resolve only strings
  containing a path separator, so a bare innocent-named link (`notes.txt` with no `/`) slipped
  through as `defer` on an MCP filesystem tool while `Write` correctly denied it. The two
  write paths now refuse the identical link. Pinned by two new tests in
  `tests/test_subversion_hook.py` (the deny plus a bare-name false-positive guard).

_0.1.0 is the first published release (on PyPI). Unreleased changes will be listed here._
