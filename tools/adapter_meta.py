"""Resolve one adapter package for the release workflows, and refuse a release that does
not name it exactly.

    python tools/adapter_meta.py --from-tag recusal-langgraph-v0.1.0
    python tools/adapter_meta.py langgraph [--tag recusal-langgraph-v0.1.0]

Both print everything a build needs as ``key=value`` lines (the format of
``$GITHUB_OUTPUT``); the first finds the adapter from a release tag. The lines are the
directory, the distribution and module names, the version, the wheel and sdist file
names, the other adapters this one requires (which a build has to install from the tree,
since none is on PyPI until it is released), and the tests that hold this adapter's gate.

With ``--tag`` the tag must be ``<distribution>-v<version>`` for the version in the tree,
or the script exits non-zero: a release whose tag and package disagree is not built.

Everything is read from the adapter's own ``pyproject.toml`` and ``__init__.py``. The one
table kept here is which test files belong to which adapter, and
``tests/test_adapter_release.py`` holds it to the files on disk.

Standard library only (``tomllib``, so Python 3.11+): it runs in the release toolchain's
environment before anything else is installed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / "adapters"

#: A release tag: the distribution name, ``-v``, and a plain three-part version.
TAG = re.compile(r"^recusal-(?P<directory>[a-z0-9]+(?:-[a-z0-9]+)*)-v(?P<version>\d+\.\d+\.\d+)$")

#: The tests that run with an adapter's framework installed, at the release commit. Each
#: adapter's own gate, audit and surface-probe tests, then the shared conformance suite
#: and the two source locks. An adapter built on another runs that one's tests too.
_SHARED = (
    "tests/test_adapter_conformance.py",
    "tests/test_adapter_packages.py",
    "tests/test_audited_routes.py",
)
_LANGGRAPH = (
    "tests/test_langgraph_gate.py",
    "tests/test_langgraph_audit.py",
    "tests/test_langgraph_interrupt.py",
    "tests/test_compat_langgraph.py",
)
_OPENAI_AGENTS = (
    "tests/test_openai_agents_gate.py",
    "tests/test_openai_agents_audit.py",
    "tests/test_compat_openai_agents.py",
)
TESTS = {
    "crewai": (
        "tests/test_crewai_gate.py",
        "tests/test_crewai_audit.py",
        "tests/test_compat_crewai.py",
    ),
    "google-adk": (
        "tests/test_google_adk_gate.py",
        "tests/test_google_adk_audit.py",
        "tests/test_compat_google_adk.py",
    ),
    "langchain": (
        *_LANGGRAPH,
        "tests/test_langchain_agent_gate.py",
        "tests/test_langchain_agent_audit.py",
        "tests/test_compat_langchain.py",
    ),
    "langgraph": _LANGGRAPH,
    "openai": (
        "tests/test_openai_responses_gate.py",
        "tests/test_openai_responses_audit.py",
    ),
    "openai-agents": _OPENAI_AGENTS,
    "openai-agents-mcp": (
        *_OPENAI_AGENTS,
        "tests/test_openai_agents_mcp_gate.py",
        "tests/test_openai_agents_mcp_audit.py",
        "tests/test_compat_openai_agents_mcp.py",
    ),
    "pydantic-ai": (
        "tests/test_pydantic_ai_gate.py",
        "tests/test_pydantic_ai_audit.py",
        "tests/test_compat_pydantic_ai.py",
    ),
}


class Refused(Exception):
    """The adapter or the tag is not one this repository can release."""


def directories() -> list[str]:
    return sorted(path.parent.name for path in ADAPTERS.glob("*/pyproject.toml"))


def directory_for_tag(tag: str) -> str:
    """The adapter directory a release tag names. Refuses anything else, including the
    kernel's own ``vX.Y.Z`` tags, which are not adapter releases."""
    match = TAG.match(tag)
    if match is None:
        raise Refused(f"{tag!r} is not an adapter release tag (recusal-<adapter>-vX.Y.Z)")
    directory = match.group("directory")
    if directory not in directories():
        raise Refused(f"{tag!r} names no adapter under adapters/ ({directory!r})")
    return directory


def resolve(directory: str, tag: str | None = None) -> dict[str, str]:
    """Everything a build needs to know about one adapter, read from its own files."""
    import tomllib

    if directory not in directories():
        raise Refused(f"no adapter at adapters/{directory}")
    project = tomllib.loads((ADAPTERS / directory / "pyproject.toml").read_text(encoding="utf-8"))
    distribution = project["project"]["name"]
    if distribution != f"recusal-{directory}":
        raise Refused(f"adapters/{directory} declares {distribution!r}, not recusal-{directory}")
    module = distribution.replace("-", "_")
    version_file = ADAPTERS / directory / project["tool"]["hatch"]["version"]["path"]
    found = re.search(r'^__version__ = "(\d+\.\d+\.\d+)"$', version_file.read_text("utf-8"), re.M)
    if found is None:
        raise Refused(f"{version_file.name} declares no plain three-part __version__")
    version = found.group(1)
    if tag is not None and tag != f"{distribution}-v{version}":
        raise Refused(
            f"tag {tag!r} does not match the package: adapters/{directory} is "
            f"{distribution} {version}, whose tag is {distribution}-v{version}"
        )
    siblings = []
    for requirement in project["project"]["dependencies"]:
        name = re.match(r"[A-Za-z0-9_.-]+", requirement).group(0)  # type: ignore[union-attr]
        if name.startswith("recusal-"):
            sibling = name[len("recusal-") :]
            if sibling not in directories():
                raise Refused(f"{distribution} requires {name}, which is not under adapters/")
            siblings.append(sibling)
    if directory not in TESTS:
        raise Refused(f"tools/adapter_meta.py lists no tests for {directory}")
    if not (ADAPTERS / directory / "tested-with.txt").is_file():
        raise Refused(f"adapters/{directory}/tested-with.txt is missing")
    return {
        "directory": directory,
        "distribution": distribution,
        "module": module,
        "version": version,
        "tag": f"{distribution}-v{version}",
        "wheel": f"{module}-{version}-py3-none-any.whl",
        "sdist": f"{module}-{version}.tar.gz",
        "siblings": " ".join(siblings),
        "tests": " ".join((*TESTS[directory], *_SHARED)),
        # the adapter's own files: with its framework installed, none of these may skip
        "own_tests": " ".join(TESTS[directory]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", nargs="?", help="the adapter's directory under adapters/")
    parser.add_argument("--tag", help="the release tag, which must match the package")
    parser.add_argument("--from-tag", help="resolve the adapter a release tag names")
    args = parser.parse_args(argv)
    try:
        if args.from_tag is not None:
            if args.directory is not None or args.tag is not None:
                parser.error("--from-tag takes no other argument")
            args.directory, args.tag = directory_for_tag(args.from_tag), args.from_tag
        elif args.directory is None:
            parser.error("name an adapter directory, or pass --from-tag")
        for key, value in resolve(args.directory, args.tag).items():
            print(f"{key}={value}")
        return 0
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
