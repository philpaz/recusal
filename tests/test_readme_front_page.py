"""What the README says at the top about what is new, held to the repository.

A "what's new" note is the part of a home page most likely to be wrong a month later. This
one names a version, and these tests fail when that version stops being the released one,
when the count of adapters changes, when it claims the kernel package is unchanged and the
packaged version says otherwise, or when the two places that say the adapters are not on
PyPI stop agreeing. The links in it are followed by ``tests/test_markdown_links.py``.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"]


def _version() -> str:
    init = (ROOT / "recusal" / "__init__.py").read_text(encoding="utf-8")
    return re.search(r'^__version__ = "([^"]+)"$', init, re.M).group(1)


def _callout() -> str:
    start = README.index("> **New on `main` since ")
    block = README[start:].split("\n\n", 1)[0]
    assert all(line.startswith(">") for line in block.splitlines()), "the note is one quote block"
    return block


def test_the_note_sits_above_the_first_section_and_below_the_badges():
    opening = README.split("\n## ", 1)[0]
    assert "> **New on `main` since " in opening
    assert opening.index("runtime%20deps") < opening.index("**Contents:**")
    assert opening.index("**Contents:**") < opening.index("> **New on `main` since ")


def test_the_note_names_the_released_version_of_the_kernel():
    """When a release is cut, the things this note calls new are released, and the note
    has to be rewritten or removed: this fails until it is."""
    version = _version()
    note = _callout()
    assert f"> **New on `main` since {version}**" in note
    assert f"`recusal` {version} on\n> PyPI is unchanged" in note
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    released = re.search(r"^## \[(\d+\.\d+\.\d+)\]", changelog, re.M).group(1)
    assert released == version, "the newest released changelog section is the packaged version"
    unreleased = changelog.split("## [Unreleased]", 1)[1].split(f"## [{version}]", 1)[0]
    assert unreleased.strip(), "the note points at an Unreleased section that has entries"


def test_the_note_counts_the_adapters_that_exist():
    adapters = sorted(p.parent.name for p in (ROOT / "adapters").glob("*/pyproject.toml"))
    assert f"**{WORDS[len(adapters)]} adapter packages**" in _callout()


def test_each_thing_the_note_calls_new_is_in_the_repository():
    note = _callout()
    for claim, evidence in (
        ("adapter packages", ROOT / "adapters" / "README.md"),
        ("hash-chained audit log", ROOT / "tests" / "test_audited_routes.py"),
        ("release path for each adapter", ROOT / ".github" / "workflows" / "release-adapter.yml"),
        ("reference architecture", ROOT / "docs" / "ENTERPRISE.md"),
    ):
        assert claim in note, claim
        assert evidence.is_file(), evidence
    # the enterprise document is described as what it is
    assert "nothing in it is built yet" in note
    enterprise = (ROOT / "docs" / "ENTERPRISE.md").read_text(encoding="utf-8")
    assert "Nothing in\nit is built, deployed or independently validated today" in enterprise


def test_every_place_that_says_the_adapters_are_not_on_pypi_says_it_together():
    """Publishing an adapter means changing all of these in one commit. While any one of
    them still says "not on PyPI", every one of them does."""
    places = {
        "README note": "The adapters are not on PyPI yet" in _callout(),
        "README section": "None is on PyPI yet" in README,
        "adapters/README.md": "**None is published to PyPI yet.**"
        in (ROOT / "adapters" / "README.md").read_text(encoding="utf-8"),
        "docs/ARCHITECTURE.md": "None has been published yet."
        in (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8"),
    }
    for directory in sorted(p.parent.name for p in (ROOT / "adapters").glob("*/pyproject.toml")):
        text = (ROOT / "adapters" / directory / "README.md").read_text(encoding="utf-8")
        places[f"adapters/{directory}/README.md"] = "not yet published to PyPI" in text
    assert len(set(places.values())) == 1, {k: v for k, v in places.items()}


def test_the_contents_line_links_sections_that_exist_in_this_order():
    line = README.split("**Contents:**", 1)[1].split("\n\n", 1)[0]
    anchors = re.findall(r"\]\(#([a-z0-9-]+)\)", line)
    assert len(anchors) == 10 and len(set(anchors)) == 10
    headings = re.findall(r"^## (.+)$", README, re.M)
    slugs = [
        re.sub(r"[^\w\- ]", "", re.sub(r"[`*_]", "", h).lower()).replace(" ", "-") for h in headings
    ]
    positions = [slugs.index(anchor) for anchor in anchors]
    assert positions == sorted(positions), "the contents line follows the page's own order"
