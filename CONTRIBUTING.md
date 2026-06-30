# Contributing to Recusal

Thanks for your interest. Recusal is deliberately small, and contributions should keep it
that way.

## Principles (please read first)

Recusal has a constitution — see [`CONSTITUTION.md`](CONSTITUTION.md). Two rules constrain
almost every change:

- **No model in the decision path.** Evidence-gathering may use a model *upstream*;
  adjudication must stay deterministic.
- **Don't grow the kernel.** New capability is a new check or a thin adapter — not a change
  to `compute_verdict`. See [`docs/EXTENDING.md`](docs/EXTENDING.md).

## Development setup

```bash
git clone https://github.com/philpaz/recusal
cd recusal
pip install -e ".[dev]"
ruff check .
pytest -q
```

Python 3.9+. **Zero runtime dependencies** — please keep `recusal/` standard-library only.

## What makes a good PR

- A new **check** (a function returning `Finding`s) or a new **adapter** (turns a `Verdict`
  into a framework's allow/deny shape).
- Tests for it. We test heavy — invariants in `tests/test_contract_invariants.py`, edge
  cases per surface. A check without edge-case tests won't merge.
- Keep findings pure (no I/O); put structured detail in `context`, not the message.

## What we'll likely decline

- A runtime dependency in the core.
- Anything that puts an LLM in the verdict path.
- Scope creep that turns the kernel into a framework.

## Code of conduct

Participation is governed by our [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) (Contributor
Covenant). Please read it before engaging.

## Reporting issues

Security-relevant reports: see [`SECURITY.md`](SECURITY.md). Everything else: open an issue
with a minimal reproduction — the templates under
[`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/) will guide you.

## License

By contributing, you agree your contributions are licensed under the project's
[Apache-2.0 License](LICENSE).
