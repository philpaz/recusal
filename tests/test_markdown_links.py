"""Every link in the repository's Markdown that points inside the repository resolves.

Three kinds of link are followed, in every tracked ``.md`` file: a relative path
(``docs/HOWTO.md``), an absolute GitHub link to this repository's main branch
(``https://github.com/philpaz/recusal/blob/main/...``, the form the README uses so it
renders on PyPI too), and a heading anchor (``#works-with``, alone or after a path). A
document that points a reader at a file or a section that is not there is the cheapest way
for careful documentation to look careless.

External links are not fetched: a test that depends on someone else's server is not a test
of this repository. Code fences and inline code are skipped, since a link shown as an
example is not a link.
"""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OWN = re.compile(r"^https://github\.com/philpaz/recusal/(?:blob|tree)/main/(?P<path>[^#?]*)")
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\((?P<target>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
HTML_ID = re.compile(r"<a\s+(?:name|id)=\"([^\"]+)\"")


def _tracked() -> list:
    listed = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return sorted(ROOT / line for line in listed.stdout.split() if line)


FILES = _tracked()


def _prose(text: str) -> str:
    """The text with fenced blocks and inline code removed."""
    text = re.sub(r"^(```|~~~).*?^\1[^\n]*$", "", text, flags=re.S | re.M)
    return re.sub(r"`[^`\n]*`", "", text)


def _slug(heading: str) -> str:
    """GitHub's anchor for a heading: lower case, punctuation dropped, spaces to hyphens.
    Inline code keeps its text and links keep their label."""
    heading = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    heading = re.sub(r"[`*_]", "", heading).strip().lower()
    heading = re.sub(r"[^\w\- ]", "", heading)
    return heading.replace(" ", "-")


def _anchors(path: Path) -> set:
    text = path.read_text(encoding="utf-8")
    without_fences = re.sub(r"^(```|~~~).*?^\1[^\n]*$", "", text, flags=re.S | re.M)
    found: dict = {}
    anchors = set(HTML_ID.findall(text))
    for line in without_fences.splitlines():
        match = re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if match is None:
            continue
        slug = _slug(match.group(1))
        count = found.get(slug, 0)
        found[slug] = count + 1
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    return anchors


def _links(path: Path) -> list:
    return [m.group("target") for m in LINK.finditer(_prose(path.read_text(encoding="utf-8")))]


def _resolve(source: Path, target: str):
    """``(file, anchor)`` for a link into this repository, or None for any other link."""
    own = OWN.match(target)
    if own is not None:
        anchor = target.split("#", 1)[1] if "#" in target else ""
        return ROOT / own.group("path").rstrip("/"), anchor
    if re.match(r"^[a-z][a-z0-9+.-]*:", target):
        return None  # another site, a mail address
    path, _, anchor = target.partition("#")
    return (source if not path else (source.parent / path)), anchor


def test_there_is_markdown_to_check():
    assert len(FILES) >= 30, len(FILES)
    assert ROOT / "README.md" in FILES and ROOT / "adapters" / "README.md" in FILES


@pytest.mark.parametrize("source", FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_link_into_the_repository_points_at_something_that_exists(source):
    broken = []
    for target in _links(source):
        resolved = _resolve(source, target)
        if resolved is None:
            continue
        path, anchor = resolved
        if not path.exists():
            broken.append(f"{target}: no such file")
        elif anchor and path.suffix == ".md" and anchor not in _anchors(path):
            broken.append(f"{target}: no heading with that anchor in {path.name}")
    assert not broken, "\n".join(broken)


def test_negative_control_a_dead_path_and_a_dead_anchor_are_both_seen(tmp_path):
    page = tmp_path / "page.md"
    page.write_text(
        "# One Heading\n\n## One Heading\n\n"
        "[ok](#one-heading) [second](#one-heading-1) [dead](#no-such) [gone](missing.md)\n"
        "`[in code](nowhere.md)`\n\n```\n[fenced](nowhere.md)\n```\n",
        encoding="utf-8",
    )
    assert _anchors(page) == {"one-heading", "one-heading-1"}
    assert _links(page) == ["#one-heading", "#one-heading-1", "#no-such", "missing.md"]
    path, anchor = _resolve(page, "#no-such")
    assert path == page and anchor not in _anchors(page)
    assert not _resolve(page, "missing.md")[0].exists()
    assert _resolve(page, "https://example.com/x.md") is None
    own = _resolve(page, "https://github.com/philpaz/recusal/blob/main/docs/HOWTO.md#x")
    assert own == (ROOT / "docs" / "HOWTO.md", "x")


def test_the_slug_follows_githubs_rule_for_the_headings_this_repository_uses():
    assert _slug("Works with") == "works-with"
    assert _slug("MCP tools, the same gate") == "mcp-tools-the-same-gate"
    assert _slug("Any agent loop, no Claude required") == "any-agent-loop-no-claude-required"
    assert _slug("Claude Code, drop-in `PreToolUse` hook") == "claude-code-drop-in-pretooluse-hook"
    assert _slug("8. Authorize one exact action (0.9.0)") == "8-authorize-one-exact-action-090"
