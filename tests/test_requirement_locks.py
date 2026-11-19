"""Every install a workflow makes from the network is hash-locked, and the locks are current.

``requirements/*.txt`` are written by ``tools/lock.py``. A job installs one with
``pip install --require-hashes``, so what it installs is the set of files that was
reviewed. This file holds three things:

- each lock is complete (every pin carries a hash) and was resolved from the roots its
  sources name today (``pyproject.toml``'s dev extra, each adapter's ``tested-with.txt``);
- every ``pip install`` in every workflow is one of the three forms that cannot fetch an
  unreviewed file, apart from the newest-release lane, which is unpinned on purpose and is
  listed here line by line;
- the forms are the ones OpenSSF Scorecard's Pinned-Dependencies check accepts, so the
  check and this test agree about what "pinned" means.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOCKS = ROOT / "requirements"
WORKFLOWS = ROOT / ".github" / "workflows"

needs_tomllib = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="tools/lock.py reads pyproject.toml with tomllib"
)


def _tool():
    spec = importlib.util.spec_from_file_location("recusal_lock_tool", ROOT / "tools" / "lock.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _locks():
    return sorted(LOCKS.glob("*.txt"))


@needs_tomllib
def test_there_is_a_lock_for_the_dev_tools_each_tool_and_each_adapter():
    tool = _tool()
    names = sorted(tool.definitions())
    adapters = sorted(p.parent.name for p in (ROOT / "adapters").glob("*/pyproject.toml"))
    assert len(adapters) == 8
    assert names == sorted(
        ["dev", "cookbook-mcp", "recusal-0.3.0", "sigstore", "zizmor"]
        + [f"adapter-{directory}" for directory in adapters]
    )
    assert sorted(path.stem for path in _locks()) == names


@needs_tomllib
def test_every_lock_was_resolved_from_the_roots_its_sources_name_today():
    """Change a pin in pyproject.toml or a tested-with.txt without relocking, and this fails
    with the command to run."""
    tool = _tool()
    assert tool.stale(sorted(tool.definitions())) == []


@needs_tomllib
def test_the_staleness_check_notices_a_changed_root(monkeypatch):
    """Negative control for the test above."""
    tool = _tool()
    monkeypatch.setattr(tool, "dev_roots", lambda: ["pytest>=8", "ruff==0.0.0"])
    problems = tool.stale(["dev"])
    assert len(problems) == 1 and "python tools/lock.py dev" in problems[0]


def test_every_pin_in_every_lock_is_exact_and_carries_a_hash():
    locks = _locks()
    assert len(locks) == 13
    for path in locks:
        lines = path.read_text(encoding="utf-8").splitlines()
        assert re.fullmatch(r"# roots: .+ \| python>=3\.\d+", lines[0]), path.name
        body = "\n".join(line for line in lines if not line.startswith("#"))
        entries = [entry for entry in re.split(r"\n(?=\S)", body) if entry.strip()]
        assert entries, path.name
        for entry in entries:
            assert re.match(r"[A-Za-z0-9_.\-]+==[^\s;]+", entry), (path.name, entry[:60])
            assert "--hash=sha256:" in entry, (path.name, entry.splitlines()[0])
        # the roots the header names are pinned in the body
        for root in lines[0].split(": ", 1)[1].split(" | ")[0].split():
            name = re.split(r"[=<>!~]", root, maxsplit=1)[0]
            assert re.search(rf"^{re.escape(name)}==", body, re.M | re.I), (path.name, name)


def _install_commands():
    """Every ``pip install`` a workflow runs: (workflow, the command's words)."""
    found = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        # join shell line continuations, drop comment lines
        text = re.sub(r"\\\n\s*", " ", text)
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            match = re.search(r"\bpip3? install\b(.*)$", line)
            if match:
                words = match.group(1).replace('"', "").split()
                found.append((path.name, words))
    return found


def _pinned(words):
    """OpenSSF Scorecard's rule (checks/raw/shell_download_validate.go, isUnpinnedPipInstall):
    an install is pinned if it requires hashes, or installs only wheel files, or is an
    editable install of a local path with ``--no-deps``."""
    if "--require-hashes" in words:
        return True
    packages = [word for word in words if not re.fullmatch(r"(--?\w+)+", word)]
    if "-e" in words or "--editable" in words:
        local = all(not re.match(r"(git|svn|hg|bzr)", package) for package in packages)
        return "--no-deps" in words and local
    return bool(packages) and all(package.endswith(".whl") for package in packages)


#: The newest-release lane resolves the newest versions on purpose: that is what it is for.
#: These are its install lines, and the only ones in any workflow that are not pinned.
UNPINNED_ON_PURPOSE = {
    "compat.yml": 4,
}


def test_every_workflow_install_is_hash_locked_except_the_newest_release_lane():
    unpinned = {}
    commands = _install_commands()
    assert len(commands) >= 50, "the parser found too few installs to be reading the workflows"
    for workflow, words in commands:
        if not _pinned(words):
            unpinned.setdefault(workflow, []).append(" ".join(words))
    assert {name: len(lines) for name, lines in unpinned.items()} == UNPINNED_ON_PURPOSE, unpinned


def test_the_pinned_rule_matches_scorecards_on_the_cases_that_matter():
    for words, expected in (
        ("--require-hashes -r requirements/dev.txt", True),
        ("--no-deps -e .", True),
        ("-e .", False),
        ("-e .[dev]", False),
        ("--quiet dist/*.whl", True),
        ("--quiet --no-deps support/*.whl dist/*.whl", True),
        ("--quiet support/*.whl dist/$WHEEL", False),
        ("zizmor==1.30.0", False),
        ("-r adapters/$DIRECTORY/tested-with.txt", False),
        ("--no-deps -e git+https://example.org/x.git@main", False),
        ("", False),
    ):
        assert _pinned(words.split()) is expected, words


def test_the_kernel_itself_is_installed_without_dependencies():
    """``recusal`` has none, and an editable install that could resolve some would be an
    install the lock does not cover."""
    for workflow, words in _install_commands():
        if "-e" in words and workflow not in UNPINNED_ON_PURPOSE:
            assert words.count("-e") == 1 and "--no-deps" in words, (workflow, words)
            assert words[words.index("-e") + 1] == ".", (workflow, words)


def test_a_wheel_installed_without_its_dependencies_is_checked_against_the_lock():
    """The wheel jobs install the built wheels with ``--no-deps`` after the lock. ``pip
    check`` then fails the job if what the wheel declares is not what the lock supplied."""
    ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    assert ci.count("/tmp/adapter/bin/pip install --quiet --no-deps dist-adapters/") == 8
    assert ci.count("/tmp/adapter/bin/pip check") == 8
    build = (WORKFLOWS / "build-adapter-dist.yml").read_text(encoding="utf-8")
    assert "/tmp/wheelcheck/bin/pip install --quiet --no-deps support/*.whl dist/*.whl" in build
    assert "/tmp/wheelcheck/bin/pip check" in build
