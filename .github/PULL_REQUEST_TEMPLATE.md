<!--
Thanks for contributing to Recusal. Keeping the kernel small and deterministic is
the whole point, please make sure your change fits the constitution before opening.
-->

## What this changes

<!-- One or two sentences. Link any issue it closes: "Closes #123". -->

## Type of change

- [ ] New check (a function returning `Finding`s)
- [ ] New adapter (turns a `Verdict` into a framework's allow/deny shape)
- [ ] New failure-taxonomy class
- [ ] Bug fix (wrong verdict / misfiring gate / crash)
- [ ] Docs / examples
- [ ] Core change (please justify, these are rare)

## Constitution checklist

<!-- See CONSTITUTION.md and CONTRIBUTING.md. PRs that miss these usually can't merge. -->

- [ ] No model call in the verdict/decision path (evidence-gathering upstream is fine).
- [ ] No new runtime dependency in `recusal/` (standard library only).
- [ ] `compute_verdict` is unchanged, **or** the change is justified above.
- [ ] Checks are pure (no I/O); structured detail goes in `context`, not the message.

## Tests

- [ ] Added/updated tests, including edge cases (a check without edge-case tests won't merge).
- [ ] `ruff check .`, `ruff format --check .`, `mypy`, and `pytest -q` all pass locally.
