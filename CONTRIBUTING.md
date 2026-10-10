# Contributing to Recusal

Thanks for your interest. Recusal is deliberately small, and contributions should keep it
that way.

## Principles (please read first)

Recusal has a constitution, see [`CONSTITUTION.md`](CONSTITUTION.md). Two rules constrain
almost every change:

- **No model in the decision path.** Evidence-gathering may use a model *upstream*;
  adjudication must stay deterministic.
- **Don't grow the kernel.** New capability is a new check or a thin adapter, not a change
  to `compute_verdict`. See [`docs/EXTENDING.md`](docs/EXTENDING.md).
- **The same evidence gives the same verdict on every supported Python.** A check that
  passes on 3.12 and fails on 3.9 for the same input breaks the determinism the whole
  project rests on. Standard-library parsers whose accepted input changed between
  versions are therefore not used in `recusal/`: `date.fromisoformat`,
  `datetime.fromisoformat` and `time.fromisoformat` accept more formats from Python 3.11
  on (`20260615`, `2026-06-15T10:00:00.5`, `2026-W24-1`). Parse against one explicit
  grammar instead, and reject everything else on every version.
  `tests/test_cross_version_determinism.py` enforces this in CI.

Before proposing anything that changes behavior, read [`STABILITY.md`](STABILITY.md): it
lists what is frozen, what each version number promises, and why a manifest schema change
is now a major release. A PR that crosses one of those lines is not rejected, but it needs
to say so in its own description.

## Development setup

```bash
git clone https://github.com/philpaz/recusal
cd recusal
pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy
pytest -q
```

Python 3.9+. **Zero runtime dependencies**, please keep `recusal/` standard-library only.

## What makes a good PR

- A new **check** (a function returning `Finding`s) or a new **adapter** (turns a `Verdict`
  into a framework's allow/deny shape).
- Tests for it. We test heavy, invariants in `tests/test_contract_invariants.py`, edge
  cases per surface. A check without edge-case tests won't merge.
- If your change touches the kernel's behavior, extend the Hypothesis property tests in
  `tests/test_kernel_properties.py` rather than only adding examples: verdict
  monotonicity, order-independence, the decision fold, fail-closed coercion, and
  fingerprint stability are generative locks, and a kernel change that cannot state its
  invariant is a kernel change we cannot review.
- Keep findings pure (no I/O); put structured detail in `context`, not the message.

## Definition of done

A PR merges only when every one of these required checks is green on its final commit.
They are the same for every contribution, large or small:

| check | what it proves |
|---|---|
| `test (ubuntu-latest, 3.9)` | the full gate on the oldest supported Python |
| `test (ubuntu-latest, 3.10)` | the full gate on 3.10 |
| `test (ubuntu-latest, 3.11)` | the full gate on 3.11 |
| `test (ubuntu-latest, 3.12)` | the full gate on 3.12 |
| `test (ubuntu-latest, 3.13)` | the full gate on 3.13 |
| `test (ubuntu-latest, 3.14)` | the full gate on the newest supported Python |
| `test (macos-latest, 3.12)` | the full gate on macOS |
| `test (windows-latest, 3.12)` | the full gate on Windows |
| `Workflow audit (zizmor)` | every workflow passes the security audit |
| `No Anthropic co-author trailer` | no commit lists an AI assistant's account as a contributor (see below) |
| `Dogfood the GitHub Action` | the shipped GitHub Action still gates correctly |
| `Action ref selects the implementation` | an Action ref runs the code at that ref |
| `Hash-locked release environment builds` | the release build is unaffected |
| `LangGraph example (Python 3.10)` | the pinned LangGraph adapter works on its minimum supported Python |
| `LangGraph example (Python 3.12)` | the pinned LangGraph adapter works on the current cross-platform smoke Python |

"The full gate" is the same four commands in every `test` job, and all four must pass:
`ruff check .`, `ruff format --check .`, `mypy`, and `pytest -q` (the whole suite, not
only the tests you touched). Run them locally before you push; they are the commands in
[Development setup](#development-setup).

**AI-assisted contributions are welcome**, judged like any other by the checks above. What
the project doesn't take is the `Co-authored-by: Claude <noreply@anthropic.com>` trailer
some coding assistants add by default: GitHub turns it into a listed repository
contributor, and this project credits people. `No Anthropic co-author trailer` refuses a
PR whose commits carry one (or an Anthropic author or committer address), and prints the
commits and the fix. A human co-author named Claude is fine; the check matches the address.
Check locally with `python tools/check_trailers.py origin/main..HEAD`.

What else a reviewer will check before merging:

- **The change is proven, not just passing.** New behavior comes with a test that fails
  without it. Reviewers will often put the old code back and confirm your test catches it.
- **Same answer on every Python.** If your change parses or compares input, it gives the
  same result on 3.9 and on the newest supported version (see Principles).
- **Documentation matches the code.** If you change behavior a document describes, update
  that document in the same PR.
- **An example that needs more than the core says so.** An example under `examples/` may
  need an optional third-party package or a newer Python than the package does. If it
  does, it must refuse clearly when the requirement is missing (a message and a non-zero
  exit, not an import error), its tests must skip with a named reason, and nothing under
  `recusal/` may import the package. The core stays standard-library only.

Workflow changes follow the existing jobs: actions pinned to full commit SHAs with the tag
in a comment, `persist-credentials: false`, and no permissions beyond what the job needs.
The zizmor audit above enforces this. `tests/test_contributing_checks.py` fails if this
table and the jobs in `.github/workflows/ci.yml` ever disagree.

## What we'll likely decline

- A runtime dependency in the core.
- Anything that puts an LLM in the verdict path.
- Scope creep that turns the kernel into a framework.

## Code of conduct

Participation is governed by our [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) (Contributor
Covenant). Please read it before engaging.

## Reporting issues

Security-relevant reports: see [`SECURITY.md`](SECURITY.md). Everything else: open an issue
with a minimal reproduction, the templates under
[`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/) will guide you.

## License

By contributing, you agree your contributions are licensed under the project's
[Apache-2.0 License](LICENSE).
