# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

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
  append-only JSONL record of every verdict; any later edit, deletion, or reorder is
  detected. Maps to OWASP Agentic logging / EU AI Act Article 12 (record-keeping).
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
