# Fuzzing

Coverage-guided fuzzing of the places where Recusal reads bytes it did not write.

| Harness | What it reads | What it holds the kernel to |
|---|---|---|
| `fuzz_audit_verify.py` | an audit log file | `verify_file` answers and never raises; `load` never raises; a file that verifies intact loads as the entries that verify |
| `fuzz_hook_event.py` | a Claude Code `PreToolUse` event on stdin | with a policy that refuses everything, the hook never raises and always emits a deny |
| `fuzz_mcp_manifest.py` | a pinned MCP manifest file | `load_manifest` returns an object or raises `ValueError`, nothing else; an accepted manifest is accepted again after a round trip |
| `fuzz_verdict.py` | a policy's findings, as JSON | `compute_verdict` returns a verdict or raises `ValueError` or `TypeError`; strict reading never passes what lenient reading refuses |
| `fuzz_deny_list.py` | a shell command a model wrote | the reference deny-list never raises, answers the same way twice, and padding a refused command does not un-refuse it |

Each harness is one file with a `TestOneInput(data: bytes)` function and the contract in
its docstring. `seeds/<harness>/` holds its starting inputs.

## Where they run

- **Every pull request and push, on every supported Python:** `tests/test_fuzz_harnesses.py`
  runs each harness on its seeds and on a few hundred deterministic mutations of them. It
  needs no fuzzing engine, so a harness cannot rot between fuzzing runs.
- **With a real engine:** `.github/workflows/fuzz.yml` builds the harnesses with
  [ClusterFuzzLite](https://google.github.io/clusterfuzzlite/) (libFuzzer through
  [atheris](https://github.com/google/atheris)) and runs them, briefly on a pull request
  that touches the kernel or a harness, and for longer each week against `main`. The build
  image is pinned by digest and the build downloads nothing: `recusal` has no runtime
  dependencies.

## Run one locally

With Docker, from the repository root:

```bash
docker build -t recusal-fuzz -f .clusterfuzzlite/Dockerfile .
docker volume create recusal-fuzz-out
docker run --rm -e FUZZING_ENGINE=libfuzzer -e SANITIZER=address -e ARCHITECTURE=x86_64 \
  -e FUZZING_LANGUAGE=python -v recusal-fuzz-out:/out recusal-fuzz compile
docker run --rm -e FUZZING_ENGINE=libfuzzer -e SANITIZER=address -e FUZZING_LANGUAGE=python \
  -e RUN_FUZZER_MODE=interactive -v recusal-fuzz-out:/out \
  gcr.io/oss-fuzz-base/base-runner run_fuzzer fuzz_audit_verify -max_total_time=60
```

Without Docker, the test file above runs the same harnesses: `pytest tests/test_fuzz_harnesses.py`.

## Adding a harness

Add `fuzz_<name>.py` with the contract in its docstring and the imports of the code under
test inside the `atheris.instrument_imports()` block (without it the run is not
coverage-guided), add at least four seeds under `seeds/fuzz_<name>/`, and add the name to
the list in `tests/test_fuzz_harnesses.py`. A crash is a finding: fix the kernel, or, if
the contract was wrong, say why in the commit that changes it.
