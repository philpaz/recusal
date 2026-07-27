# Stability and compatibility policy

Recusal is the thing empowered to refuse. A control that changes underneath you is not a
control, so this page states exactly what will not move, what a version number promises,
and what has to be true before this project calls itself 1.0.

As of **0.8.0** the classifier is `Development Status :: 4 - Beta`.

## The honest starting point

Between 2026-07-05 and 2026-07-27 this project published twenty-three releases. That was
a hardening phase driven by thirteen external review cycles, not by users asking for
changes, and it included four MCP manifest schema versions, each of which refuses its
predecessor and forces a deliberate re-pin. A reader is right to see that cadence as
instability rather than velocity.

This policy is the change. From 0.8.0 forward the rules below apply, and the first of them
is that a release now needs a reason a user would recognize.

## What is frozen

These are the surfaces a behavior-preserving change must not alter. The full technical
statement lives in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); this is the promise
attached to it.

1. **The evidence kernel.** `Finding`, `Verdict`, `Severity`, `Decision`, the
   `compute_verdict` fold (failed CRITICAL to `FAIL`, failed ERROR to `RETRY`, otherwise
   `PASS`), and the named empty-evidence intents `evaluate_policy` / `certify_evidence`.
   Finding order, check names, severities, and context keys are part of this.
2. **Manifest schema version 8.** Canonical manifest bytes and every source, instruction,
   declaration, resolved-executable, and observation-scope fingerprint.
3. **Fail-closed behavior.** A policy that raises, a malformed `PreToolUse` event, a
   missing or corrupt manifest, an unwritable audit log, and an unverifiable strict pin all
   refuse. None of these becomes permissive without a major version.
4. **CLI commands, JSON shapes, and exit codes**, including the deliberate exception that
   `recusal demo` exits 0 when it ran and adjudicates nothing.
5. **Claude Code hook semantics.** A clean verdict defers, a non-clean verdict denies, and
   the audit control identity is owned by the implementation, never by the caller.
6. **The version lock** among the Python package, the plugin, the marketplace entry, the
   vendored runtime, and the release metadata.

## What a version number promises

Recusal applies **full semantic versioning from 0.8.0 onward**, including at `0.x`. Pre-1.0
SemVer technically permits breaking changes in a minor release, and this project used that
latitude through 0.7.x. It no longer does.

| Change | Version |
|---|---|
| Any break in the frozen list above, including a manifest schema version that refuses older manifests | **MAJOR** |
| New public surface: a new check, subcommand, adapter, or optional parameter that leaves existing behavior identical | **MINOR** |
| Fixes, documentation, CI, performance, and internal refactors with no public surface change | **PATCH** |
| Dropping an end-of-life Python interpreter | **MINOR**, named in the changelog |

A security fix ships in whichever of the three fits, and says so explicitly.

## The manifest schema, specifically

The schema moved from v5 to v8 in July 2026. Each step closed a named identity gap:
runtime-name identity (v6), the resolved executable behind a launch template (v7), and the
operator-declared observation scope (v8).

**Version 8 is frozen.** If a future gap genuinely requires a v9, it is a major version,
the migration is documented before the release, and older manifests keep refusing loudly
with a migration message rather than being read under new rules. A pin that a newer Recusal
cannot verify must fail closed, never degrade quietly. That behavior is itself part of the
frozen contract.

## Deprecation

Nothing inside the frozen perimeter is removed without all three of:

1. a release that marks it deprecated in [`CHANGELOG.md`](CHANGELOG.md), naming the
   replacement;
2. a runtime warning where one is possible, emitted on **stderr**, never stdout, because in
   a `PreToolUse` hook stdout is the decision channel and a stray line there is a
   protocol-level defect, not a nuisance;
3. removal no earlier than the next major version.

Private helpers (leading underscore) are not covered, but they are still moved
deliberately, because external code can couple to anything importable.

## Release cadence

A release needs a reason a user would recognize: a fix, a named capability, or a security
issue. Hardening that no user asked for accumulates on `main` until there is a reason to
cut a version. Provenance and signing requirements are unchanged: every release is built by
the reusable builder workflow, signed at the release boundary, and independently verifiable
by the procedure in [`docs/VERIFY.md`](docs/VERIFY.md).

## Supported versions

The latest minor receives fixes. Security fixes land on the latest minor; anything older is
best-effort, and the report process is in [`SECURITY.md`](SECURITY.md).

Python **3.9 or newer**, tested in CI on 3.9 through 3.13 across Linux, macOS, and Windows.
The 3.9 floor stays until holding it forces a compromise in the enforcement code, and the
release toolchain is pinned to versions that still validate it.

## What 1.0 will mean, and what earns it

1.0 is a promise that the perimeter above is stable **in practice**, not just in intent.
This project does not schedule that; it earns it. The preconditions, stated so the claim can
be checked rather than announced:

- manifest v8 unchanged through a period of real adopter use;
- at least one deployment outside the maintainer's own machines exercising both the
  `PreToolUse` hook and MCP pinning;
- the two environment-bound validations closed by someone's real environment: the hook
  timeout authorization outcome, and the enterprise managed-settings patterns, both named as
  open in [`SECURITY.md`](SECURITY.md) and the documentation;
- two consecutive minor releases with no breaking change needed.

Until then the number stays `0.x` and the classifier stays Beta. A project whose entire
argument is that claims should be checkable does not get to make an unearned one about
itself.
