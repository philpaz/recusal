"""Two tables near the end of the README, held to the repository.

*Built to be checked* makes claims about the project itself: how many tests, on which
Pythons and systems, how many framework gates, how releases are made. Each row is checked
here against the file it cites, so the summary cannot outlive the facts. *Documentation* is
the reader's map: every document under ``docs/`` appears in it once. Links are followed by
``tests/test_markdown_links.py``.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
BLOB = "https://github.com/philpaz/recusal/blob/main/"


def _table(marker: str) -> list:
    block = README.split(f"<!-- {marker}:start -->", 1)[1].split(f"<!-- {marker}:end -->", 1)[0]
    return [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in block.strip().splitlines()[2:]
    ]


def _row(name: str) -> str:
    return next(row[1] for row in _table("built-to-be-checked") if row[0] == f"**{name}**")


# --- Built to be checked --------------------------------------------------------------------


def test_the_stated_number_of_tests_is_a_floor_the_suite_clears(request):
    stated = int(re.search(r"More than ([\d,]+),", _row("Tests")).group(1).replace(",", ""))
    assert stated % 100 == 0, "a round floor, not a count that is wrong tomorrow"
    collected_files = {item.fspath.basename for item in request.session.items}
    on_disk = {path.name for path in (ROOT / "tests").glob("test_*.py")}
    if collected_files != on_disk:
        pytest.skip("only part of the suite was collected; the floor is checked on a full run")
    assert request.session.testscollected > stated, (
        f"the README says more than {stated} tests; {request.session.testscollected} collected"
    )


def test_the_pythons_and_systems_named_are_the_ones_ci_runs():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    versions = re.search(r'python-version: \[("[\d.]+"(?:, "[\d.]+")*)\]', ci).group(1)
    listed = re.findall(r'"([\d.]+)"', versions)
    assert f"Python {listed[0]} to {listed[-1]}" in _row("Tests")
    assert listed == sorted(listed, key=lambda v: tuple(map(int, v.split("."))))
    for system, runner in (
        ("Linux", "ubuntu-"),
        ("macOS", "macos-latest"),
        ("Windows", "windows-latest"),
    ):
        assert system in _row("Tests") and runner in ci, system
    # the oldest Python named is the one the package declares
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'requires-python = ">={listed[0]}"' in project


def test_the_kernel_has_no_runtime_dependencies():
    assert _row("Runtime dependencies").startswith("None.")
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r"^dependencies = \[\]", project, re.M)


def test_the_number_of_framework_gates_and_both_test_lanes_are_real():
    words = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    adapters = sorted(p.parent.name for p in (ROOT / "adapters").glob("*/pyproject.toml"))
    assert _row("Framework gates").startswith(f"{words[len(adapters)]} packages")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    compat = (ROOT / ".github" / "workflows" / "compat.yml").read_text(encoding="utf-8")
    assert "pull_request" in ci and "pull_request" not in compat.split("jobs:", 1)[0]
    assert "schedule:" in compat, "the newest-release lane runs on a schedule"
    for directory in adapters:
        lock = f"requirements/adapter-{directory}.txt"
        roots = (ROOT / lock).read_text(encoding="utf-8").splitlines()[0].split()
        assert f"--require-hashes -r {lock}" in ci, (
            f"{lock} is not installed on the pull request lane"
        )
        for pin in (ROOT / "adapters" / directory / "tested-with.txt").read_text("utf-8").split():
            assert pin in roots, f"{pin} is not pinned on the pull request lane"
            assert pin.split("==")[0] in compat, f"{pin.split('==')[0]} is not on the weekly lane"


def test_one_conformance_suite_covers_every_gate():
    assert "One conformance suite" in _row("Every gate, one contract")
    suite = (ROOT / "tests" / "test_adapter_conformance.py").read_text(encoding="utf-8")
    examples = set(re.findall(r'"(\w+_gate\.py)"', suite))
    packaged = {
        path.name
        for path in (ROOT / "examples").glob("*_gate.py")
        if path.name not in {"allowlist_gate.py", "claude_code_gate.py", "quality_gate.py"}
    }
    assert len(packaged) == 8
    assert packaged <= examples, sorted(packaged - examples)


def test_the_release_row_describes_the_release_workflows():
    row = _row("Releases")
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    build = (ROOT / ".github" / "workflows" / "build-dist.yml").read_text(encoding="utf-8")
    verify = (ROOT / ".github" / "workflows" / "verify-release.yml").read_text(encoding="utf-8")
    assert "reusable workflow with provenance" in row
    assert "workflow_call" in build and "actions/attest-build-provenance@" in build
    assert "signed with Sigstore" in row and "sigstore/gh-action-sigstore-python@" in release
    assert "trusted publishing" in row and "pypa/gh-action-pypi-publish@" in release
    assert "negative controls" in row and verify.count("NEGATIVE CONTROL") >= 2


def test_the_rows_about_proof_and_limits_point_at_documents_that_say_so():
    proven = (ROOT / "docs" / "PROVEN.md").read_text(encoding="utf-8")
    assert "Governed by its own hook" in _row("Its own repository")
    assert "Recusal governs its own repository" in proven and "## It refuses, for real" in proven
    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert "does not defend against" in _row("Limits")
    assert "tamper-evident, not tamper-proof" in security
    # "held to the code by tests" names tests that exist
    for name in ("test_markdown_links.py", "test_adapter_map.py", "test_framework_gates.py"):
        assert (ROOT / "tests" / name).is_file(), name


def test_every_row_cites_something_to_verify_it_with():
    rows = _table("built-to-be-checked")
    assert len(rows) == 8
    for row in rows:
        assert len(row) == 3 and re.search(rf"\]\({re.escape(BLOB)}[^)]+\)", row[2]), row[0]


# --- Documentation --------------------------------------------------------------------------


def test_every_document_under_docs_is_in_the_table_exactly_once():
    linked = [
        re.search(rf"\]\({re.escape(BLOB)}([^)]+)\)", row[1]).group(1)
        for row in _table("documentation")
    ]
    assert len(linked) == len(set(linked)), "a document is listed twice"
    in_docs = sorted(
        f"docs/{path.name}" for path in (ROOT / "docs").glob("*.md") if path.name != "README.md"
    )
    assert sorted(p for p in linked if p.startswith("docs/")) == in_docs
    for path in linked:
        assert (ROOT / path).is_file(), path


def test_the_table_is_grouped_by_what_the_reader_is_trying_to_do():
    goals = [row[0].strip("*") for row in _table("documentation") if row[0]]
    assert goals == [
        "Decide whether you need it",
        "Use it",
        "Understand the design",
        "Check it yourself",
        "Depend on it",
        "Contribute",
    ]
    assert _table("documentation")[0][0], "the first row opens a group"
    # the enterprise document is described as a position wherever it is listed
    enterprise = next(row for row in _table("documentation") if "ENTERPRISE.md" in row[1])
    assert "nothing in it is built" in enterprise[2]
