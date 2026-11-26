"""Every install a workflow makes from the network is hash-locked, and the locks are current.

``requirements/*.txt`` are written by ``tools/lock.py``. A job installs one with
``pip install --require-hashes``, so what it installs is the set of files that was
reviewed. This file holds three things:

- each lock is complete (every pin carries a hash) and was resolved from the roots its
  sources name today (``pyproject.toml``'s dev extra, each adapter's ``tested-with.txt``);
- every ``pip install`` in every workflow is one of the three forms that cannot fetch a
  file whose hash was not listed first. The newest-release lane resolves the newest
  versions on purpose, and it too installs what it resolved by hash;
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
    drafts = sorted(p.parent.name for p in (ROOT / "adapters" / "drafts").glob("*/pyproject.toml"))
    assert drafts == ["llamaindex"]
    assert names == sorted(
        ["dev", "cookbook-mcp", "recusal-0.3.0", "sigstore", "uv", "zizmor"]
        + [f"adapter-{directory}" for directory in adapters]
        + [f"draft-{directory}" for directory in drafts]
    )
    assert sorted(path.stem for path in _locks()) == names


@needs_tomllib
def test_every_lock_was_resolved_from_the_roots_its_sources_name_today():
    """Change a pin in pyproject.toml or a tested-with.txt without relocking, and this fails
    with the command to run."""
    tool = _tool()
    assert tool.stale(sorted(tool.definitions())) == []


@needs_tomllib
def test_a_cooling_off_exception_is_narrow_stated_and_removed_once_it_is_not_needed():
    """A lock may take named releases before their seven days are up only to close a
    published advisory. Each exception names advisories and packages that lock pins, the
    lock says so in its header, no other lock carries such a line, and once the lock's own
    cutoff has reached the exception's date the exception closes nothing and must go."""
    tool = _tool()
    locks = tool.definitions()
    for path in _locks():
        text = path.read_text(encoding="utf-8")
        head = "".join(text.splitlines(keepends=True)[:4])
        if path.stem not in tool.COOLING_OFF_EXCEPTIONS:
            assert "# Except " not in head, path.name
            continue
        packages, until, advisories = tool.COOLING_OFF_EXCEPTIONS[path.stem]
        assert path.stem in locks
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", until)
        assert re.fullmatch(
            r"GHSA(-[23456789cfghjmpqrvwx]{4}){3}( GHSA(-[23456789cfghjmpqrvwx]{4}){3})*",
            advisories,
        )
        pinned = {name.lower() for name, _ in tool.pins(text)}
        assert set(packages) <= pinned, f"{path.name} pins none of {sorted(set(packages) - pinned)}"
        line = f"# Except {', '.join(packages)}: releases to {until}, to close {advisories}.\n"
        assert line in head, path.name
        cutoff = re.search(r"Releases newer than (\d{4}-\d{2}-\d{2})T", head).group(1)
        assert cutoff < until, (
            f"{path.name}: its cutoff {cutoff} has reached {until}; remove the exception from "
            "COOLING_OFF_EXCEPTIONS and run: python tools/lock.py " + path.stem
        )


@needs_tomllib
def test_the_staleness_check_notices_a_changed_root(monkeypatch):
    """Negative control for the test above."""
    tool = _tool()
    monkeypatch.setattr(tool, "dev_roots", lambda: ["pytest>=8", "ruff==0.0.0"])
    problems = tool.stale(["dev"])
    assert len(problems) == 1 and "python tools/lock.py dev" in problems[0]


def test_every_pin_in_every_lock_is_exact_and_carries_a_hash():
    locks = _locks()
    assert len(locks) == 15
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


#: Workflows allowed an install that is not in a pinned form: none.
UNPINNED_ON_PURPOSE: dict = {}


def test_every_workflow_install_is_hash_locked():
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


# --- the newest-release lane: resolved now, installed by hash ------------------------------


def test_the_newest_release_lane_resolves_without_pins_and_installs_what_it_resolved():
    """The lane exists to learn what the newest releases do, so it does not pin a version.
    It does pin the files: the resolver writes a lock with every hash, and that lock is what
    gets installed."""
    compat = (WORKFLOWS / "compat.yml").read_text(encoding="utf-8")
    assert compat.count("python -m pip install --require-hashes -r requirements/uv.txt") == 3
    assert compat.count("python tools/lock.py --resolve ") == 3
    assert compat.count('--require-hashes -r "$RUNNER_TEMP/newest.txt"') == 2
    assert '--require-hashes -r "$SMOKE_ROOT/framework-lock.txt"' in compat
    assert '-m pip install --no-deps "$SMOKE_ROOT/wheels/"*.whl' in compat
    assert '"$SMOKE_ROOT/venv/bin/python" -m pip check' in compat
    # and nothing in it asks the index to choose at install time
    for line in compat.splitlines():
        if "pip install" in line and not line.lstrip().startswith("#"):
            assert "--upgrade" not in line and "--constraint" not in line, line


@needs_tomllib
def test_resolve_takes_bare_names_only_so_the_lane_cannot_be_handed_a_pin():
    tool = _tool()
    for bad in ("openai==3.49.0", "openai>=3", "-r", "git+https://example.org/x.git", "a b"):
        with pytest.raises(SystemExit, match="bare distribution names"):
            tool.resolve_newest(Path("unused.txt"), [bad], with_dev=False)


def test_the_dev_roots_are_read_the_same_way_with_and_without_tomllib():
    """``--resolve --with-dev`` runs on Python 3.10, which has no tomllib, so the dev extra
    is read as text. On a Python that has tomllib, the two readings agree."""
    tool = _tool()
    roots = tool.dev_roots()
    assert roots and all(isinstance(root, str) for root in roots)
    if sys.version_info >= (3, 11):
        import tomllib

        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        assert roots == project["project"]["optional-dependencies"]["dev"]


# --- a lock can only be installed where the repository is checked out ----------------------


def _jobs(workflow: str):
    body = workflow.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([\w-]+):\n", body, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))


def test_every_job_that_installs_a_lock_or_runs_the_lock_tool_checks_the_file_out_first():
    """The two verify workflows check nothing out, on purpose: they verify published
    artifacts. Pointed at ``requirements/sigstore.txt`` they failed on a missing file, after
    a release. A job that reads a lock or a tool from the repository has a checkout before
    that step, and the path it reads is the path the checkout put it at."""
    checked = 0
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for name, body in _jobs(path.read_text(encoding="utf-8")).items():
            effective = "\n".join(
                line for line in body.splitlines() if not line.lstrip().startswith("#")
            )
            uses = re.search(r"requirements/[\w.$-]+\.txt|tools/lock\.py", effective)
            if not uses:
                continue
            checked += 1
            checkout = effective.find("uses: actions/checkout@")
            where = f"{path.name}: {name}"
            assert 0 <= checkout < uses.start(), (
                f"{where} reads {uses.group(0)} with no checkout before it"
            )
            prefix = re.search(r"^\s+path: (\S+)$", effective[checkout : uses.start()], re.M)
            if prefix:
                for found in re.finditer(r"(\S*)requirements/[\w.$-]+\.txt", effective):
                    assert found.group(1).endswith(f"{prefix.group(1)}/"), (where, found.group(0))
    assert checked >= 30, "the parser found too few jobs to be reading the workflows"


def test_the_verify_workflows_check_out_one_file_and_nothing_else():
    for name in ("verify-release.yml", "verify-adapter-release.yml"):
        workflow = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert workflow.count("uses: actions/checkout@") == 1, name
        step = workflow.split("uses: actions/checkout@", 1)[1].split("\n      - name:", 1)[0]
        assert "sparse-checkout: requirements/sigstore.txt" in step, name
        assert "sparse-checkout-cone-mode: false" in step and "path: verifier" in step, name
        assert "persist-credentials: false" in step, name
