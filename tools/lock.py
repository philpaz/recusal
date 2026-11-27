"""Write the hash-locked requirement files the workflows install from.

    python tools/lock.py                 # rewrite every lock
    python tools/lock.py dev crewai      # rewrite the named ones
    python tools/lock.py --check         # exit 1 if a lock's recorded inputs are stale
    python tools/lock.py --resolve OUT [--with-dev] PACKAGE...
                                         # the newest releases, hash-locked, for this Python

Every ``pip install`` in a workflow that reaches the network installs from one of these
files with ``--require-hashes``, so what a job installs is the exact set of files that was
resolved and listed, whatever a package index serves a moment later. Each lock is the full
resolution of a few named roots, with the sha256 of every file PyPI serves for each pinned
release.

**The committed locks** (``requirements/*.txt``) cover every Python and platform they are
used on, in one file with environment markers. Their roots are not written here twice:
they are read from where they already live, the ``dev`` extra in ``pyproject.toml`` and
each adapter's ``tested-with.txt``. The first line of each lock records the roots it was
resolved from, and ``--check`` (and ``tests/test_requirement_locks.py``) compares that line
with the sources, so a pin changed in one place and not relocked fails. A release less
than seven days old is not taken into a committed lock, the cooling-off the Dependabot
configuration uses: one that is pulled or found compromised in its first days never
enters. The one exception is ``COOLING_OFF_EXCEPTIONS``: named packages of one lock,
taken early only to close a published advisory, each named with that advisory in the
lock's own header.

**``--resolve``** is for the newest-release lane, whose purpose is to find out what the
newest releases do. It resolves the named packages with no upper pin and no cooling-off,
for the interpreter that runs it, and writes a lock to ``OUT``. The lane then installs
that file with ``--require-hashes``: what it installs is exactly what was resolved, and
the versions are on the page.

Resolution uses ``uv pip compile``. ``uv`` is installed from its own lock
(``requirements/uv.txt``) where a workflow needs it.

Standard library only. The committed locks need Python 3.11+ (``tomllib``); ``--resolve``
runs on any Python a lane uses.
"""

from __future__ import annotations

import argparse
import datetime
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKS = ROOT / "requirements"
ADAPTERS = ROOT / "adapters"
#: Draft adapters: in the tree and tested on every pull request, not released.
DRAFTS = ADAPTERS / "drafts"

#: Days a release must have been public before a committed lock may take it.
COOLING_OFF_DAYS = 7

#: Releases a lock takes before their cooling-off is over, each only to close a published
#: security advisory against what the lock held. Lock name -> (the packages, the date
#: their own cutoff moves to, the advisories it closes). Every other package in that lock
#: keeps the cooling-off. An exception the cooling-off has caught up with is stale, and a
#: test asks for it to be removed.
COOLING_OFF_EXCEPTIONS: dict[str, tuple[tuple[str, ...], str, str]] = {
    # crewai 1.15.42 made chromadb an optional extra; 1.15.40 required chromadb 1.5.9,
    # which has four advisories and no fixed release. The three are one release of CrewAI.
    "adapter-crewai": (
        ("crewai", "crewai-cli", "crewai-core"),
        "2026-11-20",
        "GHSA-36p7-vc44-83pf GHSA-f4j7-r4q5-qw2c GHSA-2wm9-hf6c-p5cr GHSA-xph7-9rjv-w5fr",
    ),
}

#: Locks with fixed roots: a tool a job installs on its own. The value is the roots and
#: the oldest Python the job that installs it runs on.
TOOLS = {
    "zizmor": (["zizmor==1.30.1"], "3.12"),
    "sigstore": (["sigstore==4.5.1"], "3.12"),
    # the resolver --resolve runs, in the newest-release lane
    "uv": (["uv==0.13.9"], "3.10"),
    # the deliberately old release the GitHub Action's dogfood job preinstalls, to prove
    # the action replaces a conflicting package
    "recusal-0.3.0": (["recusal==0.3.0"], "3.12"),
}


def _pyproject(path: Path) -> dict:
    import tomllib  # Python 3.11+; only the committed locks need it

    return tomllib.loads(path.read_text(encoding="utf-8"))


def dev_roots() -> list[str]:
    """The ``dev`` extra, read as text so that every Python a lane runs on can read it."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    line = re.search(r"^dev = \[(.+)\]$", text, re.M)
    if line is None:
        raise SystemExit("pyproject.toml: the dev extra is not on one line as `dev = [...]`")
    return re.findall(r'"([^"]+)"', line.group(1))


def kernel_python() -> str:
    project = _pyproject(ROOT / "pyproject.toml")
    return re.fullmatch(r">=(3\.\d+)", project["project"]["requires-python"]).group(1)


def adapter_directories() -> list[str]:
    return sorted(path.parent.name for path in ADAPTERS.glob("*/pyproject.toml"))


def adapter_python(directory: str) -> str:
    project = _pyproject(ADAPTERS / directory / "pyproject.toml")
    return re.fullmatch(r">=(3\.\d+)", project["project"]["requires-python"]).group(1)


def adapter_roots(directory: str) -> list[str]:
    """The dev tools plus the framework versions the adapter is tested with."""
    pins = (ADAPTERS / directory / "tested-with.txt").read_text(encoding="utf-8").split()
    return dev_roots() + pins


def draft_directories() -> list[str]:
    return sorted(path.parent.name for path in DRAFTS.glob("*/pyproject.toml"))


def draft_python(directory: str) -> str:
    project = _pyproject(DRAFTS / directory / "pyproject.toml")
    return re.fullmatch(r">=(3\.\d+)", project["project"]["requires-python"]).group(1)


def draft_roots(directory: str) -> list[str]:
    """The dev tools plus the framework versions the draft adapter is tested with."""
    pins = (DRAFTS / directory / "tested-with.txt").read_text(encoding="utf-8").split()
    return dev_roots() + pins


def definitions() -> dict[str, tuple[list[str], str]]:
    """Lock name -> (roots, oldest Python it is installed on)."""
    locks: dict[str, tuple[list[str], str]] = {"dev": (dev_roots(), kernel_python())}
    locks.update(TOOLS)
    # the cookbook job runs the documentation's MCP recipes against the pinned MCP SDK
    locks["cookbook-mcp"] = (dev_roots() + ["mcp==2.3.0"], "3.12")
    for directory in adapter_directories():
        locks[f"adapter-{directory}"] = (adapter_roots(directory), adapter_python(directory))
    for directory in draft_directories():
        locks[f"draft-{directory}"] = (draft_roots(directory), draft_python(directory))
    return locks


def header(name: str, roots: list[str], python: str) -> str:
    return f"# roots: {' '.join(roots)} | python>={python}\n"


def recorded(path: Path) -> str:
    with path.open(encoding="utf-8") as handle:
        return handle.readline()


def stale(names: list[str]) -> list[str]:
    problems = []
    locks = definitions()
    for name in names:
        roots, python = locks[name]
        path = LOCKS / f"{name}.txt"
        if not path.is_file():
            problems.append(f"{path.relative_to(ROOT).as_posix()} is missing")
        elif recorded(path) != header(name, roots, python):
            problems.append(
                f"{path.relative_to(ROOT).as_posix()} was resolved from other roots; "
                f"run: python tools/lock.py {name}"
            )
    extra = sorted(p.stem for p in LOCKS.glob("*.txt") if p.stem not in locks)
    problems += [f"requirements/{name}.txt has no definition in tools/lock.py" for name in extra]
    return problems


def resolve(roots: list[str], options: list[str], what: str) -> str:
    """The hash-locked resolution of ``roots``, as requirements text."""
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is not on PATH; install it first (pip install uv)")
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "roots.in"
        source.write_text("\n".join(roots) + "\n", encoding="utf-8")
        command = [uv, "pip", "compile", str(source), "--generate-hashes", "--no-header"]
        result = subprocess.run(
            command + ["--no-annotate", "--quiet"] + options,
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise SystemExit(f"{what}: uv could not resolve {roots}\n{result.stderr}")
    return result.stdout.replace("\r\n", "\n")


def pins(body: str) -> list[tuple[str, str]]:
    return re.findall(r"^([A-Za-z0-9_.\-]+)==([^\s;\\]+)", body, re.M)


def write(name: str, roots: list[str], python: str, cutoff: str) -> None:
    options = ["--universal", "--python-version", python, "--exclude-newer", cutoff]
    exception = ""
    if name in COOLING_OFF_EXCEPTIONS:
        packages, until, advisories = COOLING_OFF_EXCEPTIONS[name]
        for package in packages:
            options += ["--exclude-newer-package", f"{package}={until}T00:00:00Z"]
        exception = f"# Except {', '.join(packages)}: releases to {until}, to close {advisories}.\n"
    body = resolve(roots, options, name)
    note = (
        f"# Hash-locked by tools/lock.py; do not edit. Releases newer than {cutoff} were\n"
        f"# not considered. Install: python -m pip install --require-hashes -r "
        f"requirements/{name}.txt\n"
    ) + exception
    LOCKS.mkdir(exist_ok=True)
    path = LOCKS / f"{name}.txt"
    path.write_text(header(name, roots, python) + note + body, encoding="utf-8", newline="\n")
    print(f"wrote requirements/{name}.txt: {len(pins(body))} pins")


def resolve_newest(out: Path, packages: list[str], with_dev: bool) -> None:
    """Lock the newest release of each package for the running interpreter."""
    for package in packages:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-]*", package):
            raise SystemExit(f"--resolve takes bare distribution names, not {package!r}")
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    roots = (dev_roots() if with_dev else []) + packages
    body = resolve(roots, ["--python-version", python], "--resolve")
    out.write_text(
        f"# roots: {' '.join(roots)} | python=={python}, newest releases\n" + body,
        encoding="utf-8",
        newline="\n",
    )
    resolved = dict((name.lower().replace("_", "-"), version) for name, version in pins(body))
    for package in packages:
        print(f"{package}=={resolved[package.lower().replace('_', '-')]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", help="locks to rewrite, or packages with --resolve")
    parser.add_argument("--check", action="store_true", help="report stale locks and exit")
    parser.add_argument("--resolve", metavar="OUT", help="lock the newest releases to OUT")
    parser.add_argument("--with-dev", action="store_true", help="with --resolve: add the dev tools")
    args = parser.parse_args(argv)
    if args.resolve:
        if not args.names:
            parser.error("--resolve needs at least one package")
        resolve_newest(Path(args.resolve), args.names, args.with_dev)
        return 0
    if args.with_dev:
        parser.error("--with-dev is for --resolve")
    locks = definitions()
    unknown = [name for name in args.names if name not in locks]
    if unknown:
        parser.error(f"unknown lock(s) {unknown}; known: {sorted(locks)}")
    names = args.names or sorted(locks)
    if args.check:
        problems = stale(sorted(locks))
        for problem in problems:
            print(problem)
        return 1 if problems else 0
    today = datetime.datetime.now(datetime.timezone.utc).date()
    cutoff = (today - datetime.timedelta(days=COOLING_OFF_DAYS)).isoformat() + "T00:00:00Z"
    for name in names:
        write(name, *locks[name], cutoff)
    return 0


if __name__ == "__main__":
    sys.exit(main())
