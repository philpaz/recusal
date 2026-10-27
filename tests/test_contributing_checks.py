"""CONTRIBUTING.md's "Definition of done" table names exactly the checks ci.yml runs.

A contributor reads the table to learn what must pass. If a job is added, renamed or
dropped in the workflow and the table is not updated, this test fails, so the document
cannot quietly promise a different gate than the one that runs.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
GUIDE = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")


def _ci_checks():
    """Check names as GitHub reports them: `test (<os>, <python>)` per matrix cell, plus
    each other job's `name:`."""
    test_job = CI.split("\n  test:\n", 1)[1].split("\n  workflow-audit:\n", 1)[0]
    oses = re.findall(r'"?([\w.-]+)"?', re.search(r"^\s+os: \[(.*)\]$", test_job, re.M).group(1))
    pythons = re.findall(
        r'"([\d.]+)"', re.search(r"^\s+python-version: \[(.*)\]$", test_job, re.M).group(1)
    )
    cells = {(o, p) for o in oses for p in pythons}
    pair = r'^\s+- os: ([\w.-]+)\n\s+python-version: "([\d.]+)"$'
    # `exclude:` cells come before `include:` cells in the matrix; GitHub drops the
    # former and adds the latter, so the parse does the same, in that order.
    exclude, include = (
        set(re.findall(pair, part, re.M)) if part else set()
        for part in (
            test_job.split("\n        exclude:\n", 1)[1].split("\n        include:\n", 1)[0]
            if "\n        exclude:\n" in test_job
            else "",
            test_job.split("\n        include:\n", 1)[1],
        )
    )
    cells -= exclude
    cells |= include
    names = {f"test ({o}, {p})" for o, p in cells}
    names |= set(re.findall(r"^    name: (.+)$", CI, re.M))
    return names


def _guide_checks():
    section = GUIDE.split("## Definition of done", 1)[1].split("\n## ", 1)[0]
    return set(re.findall(r"^\| `([^`]+)` \|", section, re.M))


def test_ci_parse_finds_the_known_shape():
    # guards the parser itself: an empty or partial parse must not pass by accident
    checks = _ci_checks()
    assert "test (ubuntu-24.04, 3.9)" in checks
    assert "test (ubuntu-latest, 3.9)" not in checks  # the exclude: block is honored
    assert "test (ubuntu-latest, 3.10)" in checks
    assert "test (windows-latest, 3.12)" in checks
    assert "Workflow audit (zizmor)" in checks
    assert len(checks) >= 12


def test_definition_of_done_lists_every_ci_check():
    missing = _ci_checks() - _guide_checks()
    assert not missing, f"ci.yml runs checks CONTRIBUTING.md does not list: {sorted(missing)}"


def test_definition_of_done_lists_nothing_ci_does_not_run():
    stale = _guide_checks() - _ci_checks()
    assert not stale, f"CONTRIBUTING.md lists checks ci.yml does not run: {sorted(stale)}"


def test_the_full_gate_commands_match_the_test_job():
    test_job = CI.split("\n  test:\n", 1)[1].split("\n  workflow-audit:\n", 1)[0]
    section = GUIDE.split("## Definition of done", 1)[1].split("\n## ", 1)[0]
    for command in ("ruff check .", "ruff format --check .", "mypy", "pytest -q"):
        assert f"run: {command}" in test_job, f"the test job no longer runs {command!r}"
        assert f"`{command}`" in section, f"CONTRIBUTING.md does not name {command!r}"
