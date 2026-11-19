"""Write the hash-locked requirement files the workflows install from.

    python tools/lock.py                 # rewrite every lock
    python tools/lock.py dev crewai      # rewrite the named ones
    python tools/lock.py --check         # exit 1 if a lock's recorded inputs are stale

Every ``pip install`` in a workflow that reaches the network installs from one of these
files with ``--require-hashes``, so what a job installs is the exact set of files that was
reviewed, whatever a package index serves that day. Each lock is the full resolution of a
few named roots, with the sha256 of every file PyPI serves for each pinned release, for
every Python and platform the lock is used on (one file, with environment markers).

The roots are not written here twice. A lock's roots are read from where they already
live: the ``dev`` extra in ``pyproject.toml``, and each adapter's ``tested-with.txt``. The
first line of each lock records the roots it was resolved from, and ``--check`` (and
``tests/test_requirement_locks.py``) compares that line with the sources, so a pin changed
in one place and not relocked fails.

Resolution uses ``uv pip compile --universal``. ``uv`` is a tool for whoever runs this
script; nothing at install time needs it. ``--exclude-newer`` is set to seven days before
the run, the same cooling-off the Dependabot configuration uses: a release that is pulled
or found compromised in its first days never enters a lock.

Standard library only, Python 3.11+ (``tomllib``).
"""

from __future__ import annotations

import argparse
import datetime
import re
import shutil
import subprocess
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
LOCKS = ROOT / "requirements"
ADAPTERS = ROOT / "adapters"

#: Days a release must have been public before a lock may take it.
COOLING_OFF_DAYS = 7

#: Locks with fixed roots: a tool a job installs on its own. The value is the roots and
#: the oldest Python the job that installs it runs on.
TOOLS = {
    "zizmor": (["zizmor==1.30.0"], "3.12"),
    "sigstore": (["sigstore==4.4.0"], "3.12"),
    # the deliberately old release the GitHub Action's dogfood job preinstalls, to prove
    # the action replaces a conflicting package
    "recusal-0.3.0": (["recusal==0.3.0"], "3.12"),
}


def dev_roots() -> list[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return list(project["project"]["optional-dependencies"]["dev"])


def kernel_python() -> str:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return re.fullmatch(r">=(3\.\d+)", project["project"]["requires-python"]).group(1)


def adapter_directories() -> list[str]:
    return sorted(path.parent.name for path in ADAPTERS.glob("*/pyproject.toml"))


def adapter_python(directory: str) -> str:
    project = tomllib.loads((ADAPTERS / directory / "pyproject.toml").read_text(encoding="utf-8"))
    return re.fullmatch(r">=(3\.\d+)", project["project"]["requires-python"]).group(1)


def adapter_roots(directory: str) -> list[str]:
    """The dev tools plus the framework versions the adapter is tested with."""
    pins = (ADAPTERS / directory / "tested-with.txt").read_text(encoding="utf-8").split()
    return dev_roots() + pins


def definitions() -> dict[str, tuple[list[str], str]]:
    """Lock name -> (roots, oldest Python it is installed on)."""
    locks: dict[str, tuple[list[str], str]] = {"dev": (dev_roots(), kernel_python())}
    locks.update(TOOLS)
    # the cookbook job runs the documentation's MCP recipes against the pinned MCP SDK
    locks["cookbook-mcp"] = (dev_roots() + ["mcp==2.3.0"], "3.12")
    for directory in adapter_directories():
        locks[f"adapter-{directory}"] = (adapter_roots(directory), adapter_python(directory))
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


def write(name: str, roots: list[str], python: str, cutoff: str) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is not on PATH; install it to regenerate a lock (pip install uv)")
    LOCKS.mkdir(exist_ok=True)
    source = LOCKS / f".{name}.in"
    source.write_text("\n".join(roots) + "\n", encoding="utf-8")
    try:
        result = subprocess.run(
            [
                uv,
                "pip",
                "compile",
                str(source),
                "--universal",
                "--python-version",
                python,
                "--generate-hashes",
                "--no-header",
                "--no-annotate",
                "--exclude-newer",
                cutoff,
                "--quiet",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        source.unlink()
    if result.returncode != 0:
        raise SystemExit(f"{name}: uv could not resolve {roots}\n{result.stderr}")
    body = result.stdout.replace("\r\n", "\n")
    note = (
        f"# Hash-locked by tools/lock.py; do not edit. Releases newer than {cutoff} were\n"
        f"# not considered. Install: python -m pip install --require-hashes -r "
        f"requirements/{name}.txt\n"
    )
    path = LOCKS / f"{name}.txt"
    path.write_text(header(name, roots, python) + note + body, encoding="utf-8", newline="\n")
    pins = len(re.findall(r"^[A-Za-z0-9_.\-]+==", body, re.M))
    print(f"wrote requirements/{name}.txt: {pins} pins")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", help="locks to rewrite (default: all)")
    parser.add_argument("--check", action="store_true", help="report stale locks and exit")
    args = parser.parse_args(argv)
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
