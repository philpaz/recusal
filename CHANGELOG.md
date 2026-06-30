# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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

_Pre-release. No published version yet._
