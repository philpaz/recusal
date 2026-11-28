"""What the README says at the top about what is new, held to the repository.

A "what's new" note is the part of a home page most likely to be wrong a month later. This
one names a version, and these tests fail when that version stops being the released one,
when the count of adapters changes, when it claims the kernel package is unchanged and the
packaged version says otherwise, or when a page stops saying the adapters are on
PyPI. The links in it are followed by ``tests/test_markdown_links.py``.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"]


def _version() -> str:
    init = (ROOT / "recusal" / "__init__.py").read_text(encoding="utf-8")
    return re.search(r'^__version__ = "([^"]+)"$', init, re.M).group(1)


def _callout() -> str:
    start = README.index("> **New with ")
    block = README[start:].split("\n\n", 1)[0]
    assert all(line.startswith(">") for line in block.splitlines()), "the note is one quote block"
    return block


def test_the_note_sits_above_the_first_section_and_below_the_badges():
    opening = README.split("\n## ", 1)[0]
    assert "> **New with " in opening
    assert opening.index("runtime%20deps") < opening.index("**Contents:**")
    assert opening.index("**Contents:**") < opening.index("> **New with ")


def test_the_note_names_the_released_version_of_the_kernel():
    """The note names one release and links that release's changelog section. At the
    next release it names a version that is no longer the packaged one, and this fails
    until the note is rewritten or removed."""
    version = _version()
    note = _callout()
    assert f"> **New with {version}**" in note
    assert f"Under `recusal/`, {version}\n> differs from " in note
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = re.search(r"^## \[(\d+\.\d+\.\d+)\] - (\d{4}-\d{2}-\d{2})$", changelog, re.M)
    assert heading.group(1) == version, "the newest released section is the packaged version"
    anchor = f"CHANGELOG.md#{version.replace('.', '')}---{heading.group(2)})"
    assert anchor in note, "the note links the section of the release it names"
    section = changelog.split(heading.group(0), 1)[1].split("\n## [", 1)[0]
    assert re.search(r"^### (Added|Changed|Fixed|Security)$", section, re.M), (
        "the section the note links has entries"
    )


def test_the_opening_counts_the_adapters_that_exist():
    adapters = sorted(p.parent.name for p in (ROOT / "adapters").glob("*/pyproject.toml"))
    opening = README.split("\n## ", 1)[0]
    assert f"[{WORDS[len(adapters)].lower()} adapter packages]" in opening


def test_each_thing_the_note_calls_new_is_in_the_repository():
    note = _callout()
    flat = " ".join(line.lstrip("> ") for line in note.splitlines())
    for claim, evidence in (
        ("refuses a proxy and an open dispatcher", ROOT / "recusal" / "invocation.py"),
        ("This release is stricter", ROOT / "CHANGELOG.md"),
    ):
        assert claim in flat, claim
        assert evidence.is_file(), evidence
    # the refusals the note names have a public class each in the tree
    import functools
    import weakref

    from recusal import invocation

    for name, base in (
        ("ProxiedCallable", TypeError),
        ("WeakProxy", invocation.ProxiedCallable),
        ("OpenDispatch", ValueError),
    ):
        assert issubclass(getattr(invocation, name), base), name

    # the first two the note names are refused: a proxy on the way, and a single-dispatch
    # function with nothing registered, which an implementation registered after the
    # walk would change
    def pay(amount: int) -> int:
        return amount

    with pytest.raises(invocation.WeakProxy):
        invocation.reached_parameters(weakref.proxy(pay))

    @functools.singledispatch
    def charge(amount: int) -> int:
        return amount

    with pytest.raises(invocation.OpenDispatch):
        invocation.reached_parameters(charge)
    # what the note says the changelog lists is there, in the section it links
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    section = changelog.split(f"## [{_version()}]", 1)[1].split("\n## [", 1)[0]
    assert "refused although it runs correctly" in " ".join(section.lower().split())
    assert "Still not found, by design" in section
    # what the note says of the kernel diff is what the kernel diff is
    assert "in `recusal.invocation` and the version string" in flat


def test_every_place_says_the_adapters_are_on_pypi_and_installs_them_by_name():
    """All eight adapters are on PyPI. Each place that tells a reader how to get one says
    so, each adapter's README gives its own ``pip install``, and none of them has gone
    back to an install-from-repository line or to saying the package is unpublished."""
    pages = {
        "README.md": README,
        "adapters/README.md": (ROOT / "adapters" / "README.md").read_text(encoding="utf-8"),
        "docs/ARCHITECTURE.md": (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8"),
        "docs/COOKBOOK.md": (ROOT / "docs" / "COOKBOOK.md").read_text(encoding="utf-8"),
    }
    assert "> Each adapter is on PyPI under its own name and version." in _callout()
    assert "Each is on PyPI under its own name." in README
    assert "**Each is published to PyPI**" in pages["adapters/README.md"]
    assert "Each is published to PyPI." in pages["docs/ARCHITECTURE.md"]
    directories = sorted(p.parent.name for p in (ROOT / "adapters").glob("*/pyproject.toml"))
    assert len(directories) == 8
    for directory in directories:
        text = (ROOT / "adapters" / directory / "README.md").read_text(encoding="utf-8")
        pages[f"adapters/{directory}/README.md"] = text
        assert f"```bash\npip install recusal-{directory}\n```" in text, directory
    for name, text in pages.items():
        flat = " ".join(text.split())
        assert "git+https://github.com/philpaz/recusal" not in text, name
        for stale in (
            "not on PyPI",
            "not yet published",
            "published to PyPI yet",
            "None has been published",
        ):
            assert stale not in flat, (name, stale)


def test_the_contents_line_links_sections_that_exist_in_this_order():
    line = README.split("**Contents:**", 1)[1].split("\n\n", 1)[0]
    anchors = re.findall(r"\]\(#([a-z0-9-]+)\)", line)
    assert len(anchors) == 12 and len(set(anchors)) == 12
    headings = re.findall(r"^## (.+)$", README, re.M)
    slugs = [
        re.sub(r"[^\w\- ]", "", re.sub(r"[`*]", "", h).lower()).replace(" ", "-") for h in headings
    ]
    positions = [slugs.index(anchor) for anchor in anchors]
    assert positions == sorted(positions), "the contents line follows the page's own order"


def test_a_readme_that_pypi_renders_has_no_relative_link_or_image():
    """PyPI shows a package's README as its project page and resolves nothing against the
    repository, so a relative path there is a dead link or a missing picture. The kernel's
    README and each adapter's are those pages: every link and image in them is absolute."""
    # adapters/drafts/README.md is a page of the repository, not of a package on PyPI
    packages = [p for p in (ROOT / "adapters").glob("*/README.md") if p.parent.name != "drafts"]
    pages = [ROOT / "README.md", *sorted(packages)]
    assert len(pages) == 9
    for page in pages:
        text = page.read_text(encoding="utf-8")
        targets = re.findall(r'(?:src|srcset|href)="([^"]+)"', text)
        targets += re.findall(r"\]\(([^)\s]+)", text)
        assert targets, page
        relative = [t for t in targets if not t.startswith(("https://", "#", "mailto:"))]
        assert relative == [], (page.relative_to(ROOT).as_posix(), relative)


def test_each_adapter_readme_stands_on_its_own_as_a_pypi_page():
    """A reader can land on an adapter's PyPI page without having seen the kernel's. Each
    adapter README therefore says what Recusal is and what a policy is, says the package
    ships inline types (and the package does), and closes with where to go next."""
    directories = sorted(p.parent.name for p in (ROOT / "adapters").glob("*/pyproject.toml"))
    assert len(directories) == 8
    for directory in directories:
        text = (ROOT / "adapters" / directory / "README.md").read_text(encoding="utf-8")
        flat = " ".join(text.split())
        assert "**New to Recusal?**" in text, directory
        assert "No model makes that decision." in flat, directory
        assert "A policy is a plain Python function" in flat, directory
        assert "#write-the-policy-two-postures-one-principle)" in text, directory
        assert "ships inline types (`py.typed`)" in flat, directory
        module = "recusal_" + directory.replace("-", "_")
        assert (ROOT / "adapters" / directory / "src" / module / "py.typed").is_file(), directory
        more = text.split("\n## More\n", 1)
        assert len(more) == 2, directory
        for target in ("docs/FRAMEWORKS.md#", "docs/VERIFY.md#adapter-packages", "CHANGELOG.md"):
            assert target in more[1], (directory, target)
        assert more[1].rstrip().endswith("Licensed under Apache-2.0."), directory
        project = (ROOT / "adapters" / directory / "pyproject.toml").read_text(encoding="utf-8")
        assert re.search(r'^license = "Apache-2\.0"$', project, re.M), directory
    assert "## Write the policy: two postures, one principle" in README


def _kernel_picture(asset):
    """The kernel README's centered <picture> block for one asset, byte for byte."""
    found = re.findall(
        r'<p align="center">\n(?:  <a href="[^"\n]*">\n)? *<picture>\n *<source [^\n]*/assets/'
        + re.escape(asset)
        + r'\.(?:png|svg)">\n *<img [^\n]*>\n *</picture>\n(?:  </a>\n)?</p>\n',
        README,
    )
    assert len(found) == 1, asset
    return found[0]


def test_the_banner_links_to_the_website_on_every_page():
    """The banner is the first thing on the GitHub page and on each PyPI page, and it is a
    link to recusal.io. The adapter pages carry the kernel's banner block byte for byte
    (the test below), so the link cannot be on one page and missing from another."""
    banner = _kernel_picture("banner-gate-strip")
    assert README.startswith(banner)
    assert banner.startswith('<p align="center">\n  <a href="https://recusal.io">\n')
    assert banner.endswith("  </a>\n</p>\n")


def test_each_adapter_readme_shows_the_kernels_pictures_where_its_reader_needs_them():
    """An adapter's PyPI page opens like the kernel's: the same banner, then the package's
    own badges, the gate diagram beside "New to Recusal?", and the adapter map in More.
    Each picture is the kernel's block byte for byte, alternative text included, so a
    changed diagram or its description cannot leave one page behind. The python badge is
    the package's own minimum, read from its pyproject."""
    banner = _kernel_picture("banner-gate-strip")
    gate = _kernel_picture("gate-position")
    adapter_map = _kernel_picture("adapter-map")
    for asset in ("banner-gate-strip", "banner-gate-strip-light"):
        assert (ROOT / "assets" / f"{asset}.png").is_file()
    for asset in ("gate-position", "gate-position-light", "adapter-map", "adapter-map-light"):
        assert (ROOT / "assets" / f"{asset}.svg").is_file()
    directories = sorted(p.parent.name for p in (ROOT / "adapters").glob("*/pyproject.toml"))
    assert len(directories) == 8
    for directory in directories:
        package = "recusal-" + directory
        text = (ROOT / "adapters" / directory / "README.md").read_text(encoding="utf-8")
        assert text.startswith(banner + "\n# " + package + "\n"), directory
        project = (ROOT / "adapters" / directory / "pyproject.toml").read_text(encoding="utf-8")
        minimum = re.search(r'^requires-python = ">=(3\.\d+)"$', project, re.M).group(1)
        head = text.split("\n## ", 1)[0]
        for badge in (
            f"(https://img.shields.io/pypi/v/{package})](https://pypi.org/project/{package}/)",
            "(https://github.com/philpaz/recusal/actions/workflows/ci.yml/badge.svg)",
            f"(https://img.shields.io/badge/python-{minimum}%2B-blue)",
            "(https://img.shields.io/badge/license-Apache--2.0-green)",
        ):
            assert badge in head, (directory, badge)
        assert text.count(gate) == 1, directory
        assert text.index("**New to Recusal?**") < text.index(gate) < text.index("pip install"), (
            directory
        )
        more = text.split("\n## More\n", 1)[1]
        assert more.startswith("\n" + adapter_map), directory
        assert 'An arrow means "depends on". No arrow leaves the kernel' in " ".join(
            more.split()
        ), directory


def test_every_adapter_page_carries_the_website_and_scorecard_badges():
    """The adapters live in this repository, so its OpenSSF Scorecard is theirs. Each
    adapter's README shows the same website and Scorecard badges as the kernel's, so a
    reader who lands on an adapter's page (on GitHub or on PyPI) sees both."""
    website = (
        "[![Website](https://img.shields.io/badge/website-recusal.io-2453c4)](https://recusal.io)"
    )
    scorecard = (
        "[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/philpaz/recusal/badge)]"
        "(https://scorecard.dev/viewer/?uri=github.com/philpaz/recusal)"
    )
    assert README.count(website) == 1 and README.count(scorecard) == 1
    pages = sorted(p for p in (ROOT / "adapters").glob("*/README.md") if p.parent.name != "drafts")
    assert len(pages) == 8
    for page in pages:
        text = page.read_text(encoding="utf-8")
        assert text.count(website) == 1, page
        assert text.count(scorecard) == 1, page


def test_every_adapter_states_the_status_the_project_table_gives_it():
    """The kernel README's "Project status" table says where each adapter stands in security
    review, and each adapter's own page says the same in its "Status" section, with the
    invitation to help and the private route for a security gap. The table has one row per
    adapter package, so a new adapter cannot ship without a status; a status changed in one
    place and not the other fails here; and the counts in the sentence above the table are
    the table's own."""
    section = README.split("\n## Project status\n", 1)[1].split("\n## ", 1)[0]
    assert "not a finished or\ncertified security product" in section
    assert "not a third-party audit" in section
    assert "https://github.com/philpaz/recusal/security/advisories/new" in section
    row = (
        r"^\| \[`recusal-([a-z-]+)`\]\(https://github\.com/philpaz/recusal/tree/main/adapters/"
        r"([a-z-]+)\) \| (.+?) \| (.+?) \|$"
    )
    rows = re.findall(row, section, re.M)
    directories = sorted(p.parent.name for p in (ROOT / "adapters").glob("*/pyproject.toml"))
    assert [directory for _, directory, _, _ in rows] == directories
    statuses = [status for _, _, status, _ in rows]
    done, now, first = (
        statuses.count(label)
        for label in ("In-depth review complete", "In-depth review under way", "First review only")
    )
    assert done + now + first == len(rows) == 8
    sentence = section.split("Of the 8 adapters,", 1)[1].split(":", 1)[0]
    said = {
        "completed the in-depth review": done,
        "in it now": now,
        "had the first review only": first,
    }
    for phrase, expected in said.items():
        found = re.search(r"(\d+) (?:has|have|is|are) " + re.escape(phrase), sentence)
        assert (int(found.group(1)) if found else 0) == expected, phrase
    for name, directory, status, detail in rows:
        assert name == directory
        page = (ROOT / "adapters" / directory / "README.md").read_text(encoding="utf-8")
        own = page.split("\n## Status\n", 1)[1].split("\n## ", 1)[0]
        assert own.lstrip("\n").startswith(f"**{status}.** {detail} "), directory
        assert "**Help make it better.**" in own
        assert "https://github.com/philpaz/recusal/security/advisories/new" in own
        assert "https://github.com/philpaz/recusal/blob/main/CONTRIBUTING.md" in own


LIMITS = "https://github.com/philpaz/recusal/blob/main/SECURITY.md#things-worth-knowing"


def test_every_page_says_what_recusal_does_not_enforce():
    """Recusal judges only the tool calls that reach a gate. The kernel README says so near
    its top, and every adapter page says so just before its Status, each linking the limits
    in full; the section it links exists."""
    assert "\n## Things worth knowing\n" in (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    head = README.split("\n## ", 1)[0]
    assert "**What it does not enforce:**" in head
    paragraph = head.split("**What it does not enforce:**", 1)[1].split("\n\n", 1)[0]
    flat = " ".join(paragraph.split())
    assert "judges the tool calls that reach a gate you have installed" in flat
    assert "reach a resource another way" in flat and LIMITS in flat
    directories = sorted(p.parent.name for p in (ROOT / "adapters").glob("*/pyproject.toml"))
    assert len(directories) == 8
    for directory in directories:
        page = (ROOT / "adapters" / directory / "README.md").read_text(encoding="utf-8")
        before = page.split("\n## Status\n", 1)[0].rstrip()
        last = before.rsplit("\n\n", 1)[1]
        flat = " ".join(last.split())
        assert flat.startswith("**Beyond the gate.**"), directory
        assert "reach a resource another way" in flat and LIMITS in flat, directory
