# Support

Recusal is maintained by one person, with help from its contributors. Support is through
GitHub and is best effort: there are no paid plans and no guaranteed response times.

## Where to ask

- **A question, a bug or an idea:** open an
  [issue](https://github.com/philpaz/recusal/issues). For a bug, include the Recusal
  version (`python -m recusal --version`), your Python version, and the smallest input
  that shows it.
- **A security problem:** report it privately, as described in [`SECURITY.md`](SECURITY.md).
  Please don't open a public issue for it.
- **Help with a framework that isn't listed yet:** open an issue naming the framework.
  If you'd like to build the integration yourself, say so; the steps are in
  [`CONTRIBUTING.md`](CONTRIBUTING.md#adding-a-runtime).

## What is supported

- **Recusal itself:** the latest minor release receives fixes, on Python 3.9 or newer.
  The details, and what each version number promises, are in [`STABILITY.md`](STABILITY.md).
- **Framework integrations:** each runtime marked Tested in the README's
  [Works with](README.md#works-with) table, at the exact versions the README names.
  Newer framework releases are checked every week; how that works, and what happens when
  a release breaks an integration, is in [`STABILITY.md`](STABILITY.md#integrations).
- **Runtimes marked Coming or not listed** are not supported yet. Their issues say where
  the work stands.
