# Hash-locked requirements

The files here are what the workflows install from. Each is written by
[`tools/lock.py`](https://github.com/philpaz/recusal/blob/main/tools/lock.py) and installed
with `pip install --require-hashes`, so a job installs the exact files that were reviewed,
whatever a package index serves that day. Do not edit them by hand.

| File | Roots | Installed by |
|---|---|---|
| `dev.txt` | the `dev` extra in `pyproject.toml` | the test matrix, and the release gates |
| `adapter-<directory>.txt` | the dev tools, plus that adapter's `tested-with.txt` | the adapter's pull request lanes, its wheel job and its release |
| `cookbook-mcp.txt` | the dev tools, plus the pinned MCP SDK | the cookbook job |
| `zizmor.txt`, `sigstore.txt` | the workflow auditor, the Sigstore client | the audit job, the two verify workflows |
| `recusal-0.3.0.txt` | a deliberately old `recusal` | the job that proves the GitHub Action replaces a conflicting install |

`recusal` itself has no runtime dependencies, so it is installed beside a lock with
`pip install --no-deps -e .`.

One file covers every Python version and platform it is used on: a pin can carry an
environment marker, and each pin lists the hash of every file PyPI serves for it.

## Changing a pin

Change it where it lives (`pyproject.toml`, or an adapter's `tested-with.txt`), then:

```bash
pip install uv
python tools/lock.py            # or name the locks: python tools/lock.py dev adapter-crewai
```

`tests/test_requirement_locks.py` fails, with that command, if a lock no longer matches its
source. A release that is less than seven days old is not taken into a lock, the same
cooling-off the Dependabot configuration uses.

The weekly newest-release lane (`compat.yml`) is unpinned on purpose: resolving the newest
versions is what it is for.
