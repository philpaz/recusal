"""The README's table of framework gates, and the page of detail it links to.

The README used to carry every framework's full account in running prose, some 240 lines of
it. That detail is now ``docs/FRAMEWORKS.md``, one section per framework, and the README
keeps a table. These tests hold the table to the packages, the pinned versions and the
example files, hold the detail page to one section per package, and keep the long prose
from drifting back onto the home page. Links and anchors are followed by
``tests/test_markdown_links.py``.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
DETAIL = (ROOT / "docs" / "FRAMEWORKS.md").read_text(encoding="utf-8")
ADAPTERS = sorted(p.parent.name for p in (ROOT / "adapters").glob("*/pyproject.toml"))
BLOB = "https://github.com/philpaz/recusal/blob/main/"
TREE = "https://github.com/philpaz/recusal/tree/main/"


def _rows() -> dict:
    table = README.split("<!-- framework-gates:start -->", 1)[1].split(
        "<!-- framework-gates:end -->", 1
    )[0]
    rows = {}
    for line in table.strip().splitlines()[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert len(cells) == 5, line
        directory = re.search(r"\[`recusal-([a-z0-9-]+)`\]", cells[1]).group(1)
        rows[directory] = cells
    return rows


def test_the_table_has_one_row_for_every_adapter_package():
    assert sorted(_rows()) == ADAPTERS


def test_each_row_links_its_package_and_names_the_versions_a_release_installs():
    for directory, cells in _rows().items():
        assert f"({TREE}adapters/{directory})" in cells[1]
        pins = (ROOT / "adapters" / directory / "tested-with.txt").read_text("utf-8").split()
        assert re.findall(r"`([a-z0-9-]+==[0-9.]+)`", cells[2]) == pins, directory


def test_each_row_links_a_plain_example_and_an_audited_one_that_exist():
    for directory, cells in _rows().items():
        examples = re.findall(rf"\({re.escape(BLOB)}examples/(\w+\.py)\)", cells[3])
        assert len(examples) == 2, directory
        gate, audit = examples
        assert gate.endswith("_gate.py") and audit == gate.replace("_gate.py", "_audit.py")
        for name in examples:
            assert (ROOT / "examples" / name).is_file(), name


def test_each_row_links_its_own_section_of_the_detail_page():
    headings = re.findall(r"^## (.+)$", DETAIL, re.M)
    assert len(headings) == len(ADAPTERS)
    slugs = [
        re.sub(r"[^\w\- ]", "", re.sub(r"[`*]", "", h).lower()).replace(" ", "-") for h in headings
    ]
    linked = []
    for directory, cells in _rows().items():
        anchor = re.search(rf"\({re.escape(BLOB)}docs/FRAMEWORKS\.md#([a-z0-9_-]+)\)", cells[4])
        assert anchor is not None, directory
        linked.append(anchor.group(1))
        # the row's first cell is that section's heading
        assert cells[0] == headings[slugs.index(anchor.group(1))]
    assert sorted(linked) == sorted(slugs), "every section is linked from exactly one row"


def test_each_section_of_the_detail_page_names_its_package_and_its_example():
    sections = re.split(r"^## .+$", DETAIL, flags=re.M)[1:]
    assert len(sections) == len(ADAPTERS)
    named = []
    for section in sections:
        package = re.search(
            r"Package: \[`recusal-([a-z0-9-]+)`\]\(\.\./adapters/\1/README\.md\)", section
        )
        assert package is not None
        named.append(package.group(1))
        assert re.search(r"examples/\w+_gate\.py", section)
    assert sorted(named) == ADAPTERS


def test_the_long_prose_lives_on_the_detail_page_and_not_on_the_home_page():
    """One sentence from the start of each framework's account: on the detail page, and no
    longer in the README."""
    openings = [
        "For a concrete LangGraph integration,",
        "For LangChain's `create_agent` middleware,",
        "For the OpenAI Agents SDK,",
        "For the SDK's MCP tools,",
        "No framework at all, the OpenAI API called directly:",
        "For Google's Agent Development Kit,",
        "For CrewAI,",
        "For Pydantic AI,",
    ]
    assert len(openings) == len(ADAPTERS)
    for opening in openings:
        assert DETAIL.count(opening) == 1, opening
        assert opening not in README, opening


def test_no_paragraph_in_the_readmes_framework_section_is_a_wall():
    section = README.split("### Any agent loop, no Claude required\n", 1)[1].split("\n## ", 1)[0]
    prose = (
        section.split("<!-- framework-gates:start -->")[0]
        + section.split("<!-- framework-gates:end -->")[1]
    )
    longest = max(len(paragraph.splitlines()) for paragraph in prose.split("\n\n"))
    assert longest <= 12, f"a paragraph of {longest} lines is back in the README"
