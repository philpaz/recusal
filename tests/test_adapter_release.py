"""The adapter release path: one package per release, built, signed and published the way
the kernel is, and never by the kernel's own workflows.

No SDK is imported and nothing is built: these read the workflows, the resolver tool and
the adapters' files. They hold that a release names exactly one adapter at exactly the
version in the tree, that the kernel's release and an adapter's release cannot both act on
one tag, that the build is delegated to a reusable workflow whose identity a verifier can
pin, and that the verification workflow names the workflows that really build and sign.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
ADAPTERS = ROOT / "adapters"
DIRECTORIES = sorted(path.parent.name for path in ADAPTERS.glob("*/pyproject.toml"))

needs_tomllib = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="tomllib is in the standard library from Python 3.11"
)


def _workflow(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def _effective(text: str) -> str:
    """Workflow text with comment lines dropped; prose may name a key, YAML may not."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


@pytest.fixture(scope="module")
def meta():
    spec = importlib.util.spec_from_file_location(
        "adapter_meta", ROOT / "tools" / "adapter_meta.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the resolver ---------------------------------------------------------------------------


def test_there_are_adapters_to_release():
    assert len(DIRECTORIES) >= 8, DIRECTORIES


@needs_tomllib
@pytest.mark.parametrize("directory", DIRECTORIES)
def test_an_adapter_resolves_to_its_own_names_and_its_own_tag(meta, directory):
    found = meta.resolve(directory)
    assert found["distribution"] == f"recusal-{directory}"
    assert found["module"] == f"recusal_{directory.replace('-', '_')}"
    assert found["tag"] == f"recusal-{directory}-v{found['version']}"
    assert found["wheel"] == f"{found['module']}-{found['version']}-py3-none-any.whl"
    assert found["sdist"] == f"{found['module']}-{found['version']}.tar.gz"
    assert (ADAPTERS / directory / "src" / found["module"] / "__init__.py").is_file()
    # the tag round-trips: the tag names this directory, and matches this version
    assert meta.directory_for_tag(found["tag"]) == directory
    assert meta.resolve(directory, found["tag"]) == found


@needs_tomllib
@pytest.mark.parametrize("directory", DIRECTORIES)
def test_a_tag_for_another_version_or_another_adapter_is_refused(meta, directory):
    found = meta.resolve(directory)
    with pytest.raises(meta.Refused, match="does not match the package"):
        meta.resolve(directory, f"recusal-{directory}-v99.0.0")
    other = next(d for d in DIRECTORIES if d != directory)
    with pytest.raises(meta.Refused, match="does not match the package"):
        meta.resolve(directory, f"recusal-{other}-v{found['version']}")


@pytest.mark.parametrize(
    "tag",
    [
        "v0.14.0",  # the kernel's own tag
        "recusal-v0.14.0",
        "recusal-langgraph-0.1.0",  # no -v
        "recusal-langgraph-v0.1",
        "recusal-langgraph-v0.1.0rc1",
        "recusal-nope-v0.1.0",
        "recusal-langgraph-v0.1.0; rm -rf /",
        "recusal-../langgraph-v0.1.0",
        "",
    ],
)
def test_a_tag_that_names_no_adapter_release_is_refused(meta, tag):
    with pytest.raises(meta.Refused):
        meta.directory_for_tag(tag)


@needs_tomllib
def test_the_command_line_fails_closed(meta, capsys):
    assert meta.main(["--from-tag", "v0.14.0"]) == 1
    assert meta.main(["langgraph", "--tag", "recusal-langgraph-v99.0.0"]) == 1
    assert meta.main(["nope"]) == 1
    out = capsys.readouterr()
    assert "refused:" in out.err and "directory=" not in out.out
    assert meta.main(["--from-tag", meta.resolve("langgraph")["tag"]]) == 0
    assert "directory=langgraph" in capsys.readouterr().out


@needs_tomllib
def test_an_adapter_that_requires_another_names_it_as_a_sibling(meta):
    assert meta.resolve("langchain")["siblings"] == "langgraph"
    assert meta.resolve("openai-agents-mcp")["siblings"] == "openai-agents"
    assert meta.resolve("langgraph")["siblings"] == ""


def test_every_adapter_has_its_tests_listed_and_every_listed_test_exists(meta):
    assert sorted(meta.TESTS) == DIRECTORIES
    for directory, tests in meta.TESTS.items():
        assert len(set(tests)) == len(tests), directory
        for test in (*tests, *meta._SHARED):
            assert (ROOT / test).is_file(), test


def test_each_adapters_listed_tests_include_its_own_gate_and_audit_tests(meta):
    """The example a package came from names its tests: ``examples/<stem>_gate.py`` is
    held by ``tests/test_<stem>_gate.py`` and ``tests/test_<stem>_audit.py``."""
    sys.path.insert(0, str(ROOT / "tests"))
    try:
        routes = importlib.import_module("test_audited_routes")
    finally:
        sys.path.pop(0)
    by_directory = {source[0].split("/")[1]: gate for gate, source in routes.PACKAGED.items()}
    assert sorted(by_directory) == DIRECTORIES
    for directory, gate in by_directory.items():
        stem = gate[: -len("_gate.py")]
        for wanted in (f"tests/test_{stem}_gate.py", f"tests/test_{stem}_audit.py"):
            assert wanted in meta.TESTS[directory], f"{directory}: {wanted}"


@pytest.mark.parametrize("directory", DIRECTORIES)
def test_tested_with_names_the_pins_the_pull_request_lane_installs(directory):
    """One set of pins: what a release installs for an adapter is what its wheel job in
    ci.yml installs on every pull request."""
    pins = (ADAPTERS / directory / "tested-with.txt").read_text(encoding="utf-8").split()
    assert pins and all(re.fullmatch(r"[a-z0-9-]+==[0-9.]+", pin) for pin in pins), pins
    ci = _workflow("ci.yml")
    job = ci.split(f"  adapter-{directory}-wheel:\n", 1)[1].split("\n  # ", 1)[0]
    install = next(line for line in job.splitlines() if "/tmp/adapter/bin/pip install" in line)
    assert re.findall(r"[a-z0-9-]+==[0-9.]+", install) == pins


@needs_tomllib
def test_the_adapters_readme_lists_every_package_and_what_it_requires(meta):
    text = (ADAPTERS / "README.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| `([a-z0-9-]+)` \| `([a-z0-9-]+)` \| (.*?) ?\|$", text, re.M)
    documented = {directory: (package, requires.strip()) for directory, package, requires in rows}
    assert sorted(documented) == DIRECTORIES
    for directory, (package, requires) in documented.items():
        found = meta.resolve(directory)
        assert package == found["distribution"]
        wanted = " ".join(f"`recusal-{s}`" for s in found["siblings"].split())
        assert requires == wanted, f"{directory}: {requires!r}"
    # the publisher settings a maintainer copies are the workflow and environment in use
    release = _workflow("release-adapter.yml")
    assert "| Workflow name | `release-adapter.yml` |" in text
    assert "| Environment name | `pypi` |" in text and "      name: pypi\n" in release


# --- one tag, one workflow ------------------------------------------------------------------


def test_the_kernels_release_and_an_adapters_release_never_act_on_the_same_tag(meta):
    kernel, adapter = (
        _effective(_workflow("release.yml")),
        _effective(_workflow("release-adapter.yml")),
    )
    assert (
        "if: github.event_name == 'workflow_dispatch' || "
        "startsWith(github.event.release.tag_name, 'v')" in kernel
    )
    assert (
        "if: github.event_name == 'workflow_dispatch' || "
        "startsWith(github.event.release.tag_name, 'recusal-')" in adapter
    )
    # the guard sits on the first job of each, and every later job needs it
    assert kernel.index("startsWith(github.event.release.tag_name") < kernel.index("  build:")
    assert adapter.index("startsWith(github.event.release.tag_name") < adapter.index("  test:")
    for job in ("test", "build", "publish"):
        block = adapter.split(f"\n  {job}:\n", 1)[1].split("\n  sign:\n", 1)[0]
        assert re.search(r"needs:\s*(resolve|\[resolve,)", block), f"{job} does not need resolve"
    # no tag satisfies both prefixes, and an adapter tag always carries the adapter prefix
    assert not "recusal-".startswith("v") and meta.TAG.pattern.startswith("^recusal-")


def test_both_workflows_offer_exactly_the_adapters_on_disk():
    for name in ("release-adapter.yml", "verify-adapter-release.yml"):
        text = _workflow(name)
        options = text.split("        options:\n", 1)[1]
        listed = []
        for line in options.splitlines():
            found = re.fullmatch(r"          - ([a-z0-9-]+)", line)
            if found is None:
                break
            listed.append(found.group(1))
        assert listed == DIRECTORIES, f"{name}: {listed}"


def test_a_refusal_from_the_resolver_cannot_be_hidden_by_a_pipe():
    """A step's default shell reports the status of the last command in a pipe, so piping
    the resolver into anything would let a refused tag through."""
    for name in ("release-adapter.yml", "build-adapter-dist.yml"):
        for line in _effective(_workflow(name)).splitlines():
            if "adapter_meta.py" in line:
                assert "|" not in line and "$(" not in line, f"{name}: {line.strip()}"
    for line in _effective(_workflow("release-adapter.yml")).splitlines():
        if line.strip().startswith("pytest"):
            assert "|" not in line, line.strip()


def test_untrusted_text_reaches_scripts_as_environment_never_as_template():
    """An input or a tag interpolated into a ``run:`` script is a script injection; they
    are passed through ``env:`` and read as shell variables."""
    for name in ("release-adapter.yml", "build-adapter-dist.yml", "verify-adapter-release.yml"):
        in_run = False
        for line in _effective(_workflow(name)).splitlines():
            stripped = line.strip()
            if re.match(r"(- )?run: ", stripped):
                in_run = stripped.endswith("|")
                assert "${{" not in stripped, f"{name}: {stripped}"
            elif in_run and re.match(r"(- |[a-z-]+: )", stripped) and not line.startswith(" " * 10):
                in_run = False
            elif in_run:
                assert "${{" not in stripped, f"{name}: {stripped}"


# --- the build, the signature, the publish -------------------------------------------------


def test_the_adapter_release_delegates_the_build_to_its_own_reusable_builder():
    release = _workflow("release-adapter.yml")
    assert "uses: ./.github/workflows/build-adapter-dist.yml" in release
    assert "actions/attest-build-provenance@" not in release
    # the kernel's builder is not reused: its identity is the one every published kernel
    # verification command names, and it stays the kernel's alone
    assert "build-dist.yml" not in _effective(release)
    build = _workflow("build-adapter-dist.yml")
    assert "workflow_call" in build
    assert "--require-hashes -r release-requirements.txt" in build
    assert 'build --no-isolation --outdir dist "adapters/$DIRECTORY"' in build
    assert "twine check dist/*" in build
    assert "anchore/sbom-action@" in build and "upload-release-assets: false" in build
    assert "actions/attest-build-provenance@" in build and "subject-path: dist/*" in build
    assert "smoke_test.py" in build and "tested-with.txt" in build


def test_what_is_built_beside_the_adapter_is_never_attested_or_published():
    """The kernel and a required adapter are built for the smoke test into ``support/``.
    Only ``dist/`` is attested, uploaded, signed and published."""
    build = _effective(_workflow("build-adapter-dist.yml"))
    assert "--outdir support ." in build and '--outdir support "adapters/$sibling"' in build
    assert "subject-path: dist/*" in build and "path: dist/" in build
    assert not re.search(r"(subject-path|path): support", build)
    release = _effective(_workflow("release-adapter.yml"))
    assert "support" not in release


def test_the_adapter_release_is_signed_and_publishes_only_behind_the_signature():
    release = _workflow("release-adapter.yml")
    assert "sigstore/gh-action-sigstore-python@" in release
    assert "release-signing-artifacts: true" in release
    assert re.search(r"needs:\s*\[resolve,\s*build,\s*sign\]", release)
    assert "pypa/gh-action-pypi-publish@" in release
    assert "skip-existing" not in _effective(release)
    # signing and publishing happen on a published release only, never on a rehearsal
    assert release.count("if: github.event_name == 'release'") == 2


def test_the_release_gate_runs_the_whole_suite_and_then_the_adapter_with_its_framework():
    release = _effective(_workflow("release-adapter.yml"))
    test = release.split("\n  test:\n", 1)[1].split("\n  build:\n", 1)[0]
    for step in ("ruff check .", "ruff format --check .", "run: mypy", "run: pytest -q\n"):
        assert step in test, step
    assert 'pip install -r "adapters/$DIRECTORY/tested-with.txt"' in test
    assert "pytest -q $TESTS" in test and "pytest -q -rs $OWN_TESTS" in test
    assert 'grep -q "^SKIPPED"' in test
    build = release.split("\n  build:\n", 1)[1].split("\n  sign:\n", 1)[0]
    assert re.search(r"needs:\s*\[resolve,\s*test\]", build), "the build must wait for the gate"


# --- verification ---------------------------------------------------------------------------


def test_the_verifier_names_the_workflows_that_really_build_and_sign():
    verify = _workflow("verify-adapter-release.yml")
    assert (WORKFLOWS / "build-adapter-dist.yml").is_file()
    assert '--signer-workflow "${REPO}/.github/workflows/build-adapter-dist.yml"' in verify
    assert ".github/workflows/release-adapter.yml@refs/tags/${TAG}" in verify
    assert 'echo "TAG=${DISTRIBUTION}-v${VERSION}"' in verify
    assert 'publisher.get("workflow") != "release-adapter.yml"' in verify
    # verification works from the published artifacts alone
    assert "actions/checkout@" not in verify


def test_the_verifiers_negative_controls_are_the_kernels_own_workflows():
    """The wrong answers offered are real workflows in this repository that did not build
    or sign an adapter, so a verifier that accepted "any workflow here" would fail."""
    verify = _workflow("verify-adapter-release.yml")
    assert verify.count("NEGATIVE CONTROL") == 2
    assert '--signer-workflow "${REPO}/.github/workflows/build-dist.yml"' in verify
    assert ".github/workflows/release.yml@refs/tags/${TAG}" in verify
    assert verify.count("refused, as it must") == 2


def test_both_verifiers_pin_the_same_sigstore_client():
    pins = {
        name: re.search(r'pip install "sigstore==([0-9.]+)"', _workflow(name)).group(1)
        for name in ("verify-release.yml", "verify-adapter-release.yml")
    }
    assert len(set(pins.values())) == 1, pins
