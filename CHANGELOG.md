# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

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
