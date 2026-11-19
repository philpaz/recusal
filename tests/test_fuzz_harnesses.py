"""The fuzz harnesses under fuzz/ run in the ordinary suite, on their seeds and on mutations
of them, so a harness cannot rot between fuzzing runs and its contract is checked on every
Python the kernel supports.

The coverage-guided runs are ClusterFuzzLite's (.github/workflows/fuzz.yml): they build
these same files with atheris and libFuzzer. This file needs neither. What is held here:

- every harness holds its contract on every seed and on a few hundred deterministic
  mutations of them;
- every harness has seeds, and every seed folder has a harness;
- the build is pinned and downloads nothing, and the workflow builds and runs the harnesses.
"""

import importlib.util
import random
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FUZZ = ROOT / "fuzz"
HARNESSES = sorted(path.stem for path in FUZZ.glob("fuzz_*.py"))


def _load(name):
    spec = importlib.util.spec_from_file_location(f"recusal_fuzz_{name}", FUZZ / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seeds(name):
    return sorted((FUZZ / "seeds" / name).iterdir())


def _mutations(seeds, count):
    """Deterministic byte-level mutations: the kinds a fuzzer makes first."""
    rng = random.Random(20261119)
    extras = [b"\n", b"\x00", b"\xff\xfe", b"{", b"}", b'"', b"\\u0000", b"\r\n", b"9" * 40]
    for index in range(count):
        if index % 6 == 0:
            yield bytes(rng.randrange(256) for _ in range(rng.randint(0, 60)))
            continue
        data = bytearray(rng.choice(seeds))
        for _ in range(rng.randint(1, 4)):
            kind = rng.randint(0, 5)
            if kind == 0 and data:
                data[rng.randrange(len(data))] = rng.randrange(256)
            elif kind == 1:
                data.insert(rng.randint(0, len(data)), rng.randrange(256))
            elif kind == 2 and data:
                del data[rng.randrange(len(data))]
            elif kind == 3 and data:
                at = rng.randrange(len(data))
                data[at:at] = data[at : at + rng.randint(1, 20)]
            elif kind == 4:
                data += rng.choice(extras)
            else:
                del data[rng.randint(0, len(data)) :]
        yield bytes(data)


def test_there_are_harnesses_and_each_names_its_contract():
    assert HARNESSES == [
        "fuzz_audit_verify",
        "fuzz_deny_list",
        "fuzz_hook_event",
        "fuzz_mcp_manifest",
        "fuzz_verdict",
    ]
    for name in HARNESSES:
        source = (FUZZ / f"{name}.py").read_text(encoding="utf-8")
        assert source.startswith('"""Fuzz '), name
        assert "def TestOneInput(data: bytes) -> None:" in source, name
        assert "atheris.Setup(sys.argv, TestOneInput)" in source, name
        # without this the engine has no coverage to follow and the run is random bytes
        block = source.split("with atheris.instrument_imports() if atheris else", 1)
        assert len(block) == 2, name
        instrumented = block[1].split("def ", 1)[0]
        assert "from recusal" in instrumented or "import recusal" in instrumented, name


def test_every_harness_has_seeds_and_every_seed_folder_has_a_harness():
    assert sorted(path.name for path in (FUZZ / "seeds").iterdir()) == HARNESSES
    for name in HARNESSES:
        assert len(_seeds(name)) >= 4, name


@pytest.mark.parametrize("name", HARNESSES)
def test_the_harness_holds_its_contract_on_its_seeds_and_on_mutations_of_them(name):
    harness = _load(name)
    seeds = [path.read_bytes() for path in _seeds(name)]
    for data in seeds:
        harness.TestOneInput(data)
    for data in _mutations(seeds, 400):
        harness.TestOneInput(data)


def test_a_harness_fails_when_its_contract_is_broken():
    """Negative control: the same harness, with its target made to misbehave, raises. A
    harness that could not fail would make every green fuzzing run meaningless."""
    harness = _load("fuzz_audit_verify")
    seed = (FUZZ / "seeds" / "fuzz_audit_verify" / "intact.jsonl").read_bytes()
    harness.TestOneInput(seed)
    real = harness.audit.verify_file
    harness.audit.verify_file = lambda path: (True, ["a problem, and yet intact"])
    try:
        with pytest.raises(AssertionError):
            harness.TestOneInput(seed)
    finally:
        harness.audit.verify_file = real

    hook = _load("fuzz_hook_event")
    hook.run_pretooluse_hook = lambda policy, **kwargs: None  # a hook that defers
    with pytest.raises(AssertionError, match="the hook deferred"):
        hook.TestOneInput(b"{}")


def test_the_fuzzing_build_is_pinned_and_downloads_nothing():
    dockerfile = (ROOT / ".clusterfuzzlite" / "Dockerfile").read_text(encoding="utf-8")
    froms = re.findall(r"^FROM (\S+)$", dockerfile, re.M)
    assert len(froms) == 1
    assert re.fullmatch(r"gcr\.io/oss-fuzz-base/base-builder-python@sha256:[0-9a-f]{64}", froms[0])
    build = (ROOT / ".clusterfuzzlite" / "build.sh").read_text(encoding="utf-8")
    code = "\n".join(line for line in build.splitlines() if not line.lstrip().startswith("#"))
    for fetcher in ("pip", "curl", "wget", "git clone", "apt"):
        assert fetcher not in code, fetcher
    assert 'compile_python_fuzzer "$harness"' in code and "fuzz_*.py" in code
    project = (ROOT / ".clusterfuzzlite" / "project.yaml").read_text(encoding="utf-8")
    assert project.strip() == "language: python"


def test_the_workflow_builds_and_runs_the_harnesses_with_pinned_actions():
    workflow = (ROOT / ".github" / "workflows" / "fuzz.yml").read_text(encoding="utf-8")
    uses = re.findall(r"uses: (\S+)", workflow)
    assert [use.split("@")[0] for use in uses] == [
        "google/clusterfuzzlite/actions/build_fuzzers",
        "google/clusterfuzzlite/actions/run_fuzzers",
    ]
    for use in uses:
        assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", use), use
    assert workflow.count("language: python") == 2
    assert "pull_request:" in workflow and "schedule:" in workflow
    assert "permissions: read-all" in workflow
    # a pull request that touches the kernel or a harness is fuzzed
    for path in ('"recusal/**"', '"fuzz/**"'):
        assert path in workflow, path
