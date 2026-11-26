# Draft adapters

An adapter in this folder is a draft. It is in the tree so that it can be read, tried and
improved in the open, and its own tests run on every pull request with its framework
installed from a hash-locked file. It is **not released**: it is not on PyPI, the release
workflows do not offer it, and it is not counted among the adapter packages in
[`adapters/README.md`](../README.md).

A draft has not been through the review rounds the released adapters have
([`docs/REVIEW-LOG.md`](../../docs/REVIEW-LOG.md)). Read its own README for what it
covers and what it does not.

| Draft | Framework | Tested with | Contributed by |
|---|---|---|---|
| [`llamaindex`](llamaindex/README.md) | LlamaIndex `FunctionTool` | `llama-index-core==0.14.25` | @Varma0604 (#311) |

## What a draft has

- its own `pyproject.toml`, source and README, and the README says in its first heading
  that it is a draft and, in its text, that it is not published;
- a `tested-with.txt` naming the framework versions its tests run against, and a lock
  written from it (`python tools/lock.py draft-<directory>`);
- a job in `ci.yml` that installs that lock and runs the draft's tests, on the draft's
  minimum Python and on Python 3.12.

`tests/test_draft_adapters.py` holds each of these, and holds that no release workflow
names a draft.

## From draft to adapter

A draft moves to `adapters/<directory>/` when it has what every released adapter has: the
shared conformance suite, a decision on its audit route, a wheel job, a smoke test, and
review rounds recorded in the review log. Help with any of these is very welcome.
