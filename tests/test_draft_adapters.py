"""A draft adapter is tested on every pull request and is never offered for release.

``adapters/drafts/<directory>/`` holds an adapter that is in the tree to be read, tried and
improved, and that has not been through the reviews a released adapter has. This file
holds what ``adapters/drafts/README.md`` says about one:

- it says it is a draft, in its README's first heading and in its text;
- its tests run in ``ci.yml`` with its framework installed from its own hash lock, and
  the lock is written from its ``tested-with.txt``;
- no release workflow names it, and the adapter packages do not count it.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRAFTS = ROOT / "adapters" / "drafts"
WORKFLOWS = ROOT / ".github" / "workflows"
DIRECTORIES = sorted(path.parent.name for path in DRAFTS.glob("*/pyproject.toml"))

needs_tomllib = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="pyproject.toml is read with tomllib"
)


def _project(directory):
    import tomllib

    text = (DRAFTS / directory / "pyproject.toml").read_text(encoding="utf-8")
    return tomllib.loads(text)["project"]


def _jobs(directory):
    """{Python version: the text of the draft's job for it in ci.yml}."""
    ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    pattern = rf"^  draft-{re.escape(directory)}-py3\d+:\n(?:(?:    .*)?\n)+"
    jobs = {}
    for found in re.finditer(pattern, ci, re.M):
        text = found.group(0)
        jobs[re.search(r'python-version: "([0-9.]+)"', text).group(1)] = text
    assert jobs, f"ci.yml has no job for the draft {directory}"
    return jobs


def test_every_folder_of_drafts_is_a_draft_and_the_readme_lists_it():
    folders = sorted(path.name for path in DRAFTS.iterdir() if path.is_dir())
    assert folders == DIRECTORIES
    listing = (DRAFTS / "README.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| \[`([a-z0-9-]+)`\]\(\1/README\.md\) \|", listing, re.M)
    assert rows == DIRECTORIES


@pytest.mark.parametrize("directory", DIRECTORIES)
def test_a_draft_says_it_is_a_draft_and_that_it_is_not_published(directory):
    readme = (DRAFTS / directory / "README.md").read_text(encoding="utf-8")
    assert "(draft)" in readme.splitlines()[0]
    assert re.search(r"not\s+yet\s+been\s+accepted\s+or\s+published", readme)


@needs_tomllib
@pytest.mark.parametrize("directory", DIRECTORIES)
def test_a_draft_names_its_package_and_bounds_what_it_requires(directory):
    project = _project(directory)
    assert project["name"] == f"recusal-{directory}"
    assert project["license"] == "Apache-2.0"
    for requirement in project["dependencies"]:
        assert ">=" in requirement and "<" in requirement, f"{requirement} is not bounded"


@pytest.mark.parametrize("directory", DIRECTORIES)
def test_the_listing_names_the_versions_the_draft_is_tested_with(directory):
    pins = (DRAFTS / directory / "tested-with.txt").read_text(encoding="utf-8").split()
    assert pins and all(re.fullmatch(r"[a-z0-9-]+==[0-9.]+", pin) for pin in pins)
    listing = (DRAFTS / "README.md").read_text(encoding="utf-8")
    row = next(line for line in listing.splitlines() if line.startswith(f"| [`{directory}`]"))
    for pin in pins:
        assert f"`{pin}`" in row


@pytest.mark.parametrize("directory", DIRECTORIES)
def test_the_drafts_tests_run_on_every_pull_request_from_its_own_lock(directory):
    assert (ROOT / "requirements" / f"draft-{directory}.txt").is_file()
    tests = sorted(
        path.relative_to(ROOT).as_posix() for path in (DRAFTS / directory).glob("test_*.py")
    )
    assert tests
    for job in _jobs(directory).values():
        installs = [line.strip() for line in job.splitlines() if "pip install" in line]
        assert installs == [
            f"python -m pip install --require-hashes -r requirements/draft-{directory}.txt",
            "python -m pip install --no-deps -e .",
        ]
        for test in tests:
            assert test in job, f"ci.yml does not run {test}"


@needs_tomllib
@pytest.mark.parametrize("directory", DIRECTORIES)
def test_the_job_runs_on_the_drafts_minimum_python_and_on_312(directory):
    minimum = re.fullmatch(r">=(3\.\d+)", _project(directory)["requires-python"]).group(1)
    assert sorted(_jobs(directory)) == sorted({minimum, "3.12"})


@pytest.mark.parametrize("directory", DIRECTORIES)
def test_no_release_workflow_names_a_draft(directory):
    for name in ("release-adapter.yml", "verify-adapter-release.yml", "release.yml"):
        text = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert directory not in text, f"{name} names the draft {directory}"


def test_a_draft_is_not_counted_among_the_adapter_packages():
    released = {p.parent.name for p in (ROOT / "adapters").glob("*/pyproject.toml")}
    assert not released & set(DIRECTORIES)
    assert "drafts" not in released
