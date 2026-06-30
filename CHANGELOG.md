# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Evidence contract** — `Finding`, `Verdict`, `Severity`, `Decision`, and
  `compute_verdict` (the typed, zero-dependency spine). See `docs/EVIDENCE.md`.
- **Built-in checks** — `row_count`, `null_rate`, `referential_integrity`, `in_set`,
  `in_range`, `required_keys` (operate on any dict-like rows; no pandas required).
- **`GateAdjudicator`** — staged `G0`–`G8` release checkpoints and a release-evidence rollup.
- **Claude adapters** — `recusal.claude` (manual-loop tool gate + Managed Agents
  confirmation) and `recusal.claude_code` (a `PreToolUse` hook that denies even under
  `bypassPermissions` and fails closed on a policy error).
- **Tamper-evident audit log** — `recusal.audit` (`AuditLog`, `verify`): a hash-chained,
  append-only JSONL record of every verdict; any later edit, deletion, or reorder is
  detected. Maps to OWASP Agentic logging / EU AI Act Article 14.
- **Dogfood** — Recusal governs its own repository via a real Claude Code hook; verbatim,
  reproducible, CI-locked proof in `docs/PROVEN.md`.
- **Examples** — offline refusal demo, live Claude-agent demo, an OWASP-mapped scenario
  gallery, a Claude Code hook, and an audit-log demo.
- **Docs** — `CONSTITUTION`, `docs/WHY`, `docs/EVIDENCE`, `docs/HOWTO`, `docs/EXTENDING`,
  `docs/LANDSCAPE`, `docs/PROVEN`.
- **Tooling** — zero runtime dependencies; ruff (lint + format), mypy, pytest, and
  pre-commit, all run in CI.

_Pre-release. No published version yet._
