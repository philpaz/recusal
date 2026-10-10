"""The README's "Works with" table claims only what the repository proves.

A runtime marked Tested must link to something that exists: an example file that a test
in this repository exercises, or a README section. A runtime that is not tested yet must
be marked Coming and link the issue or pull request where the work happens. And no
document may say the gate drops into a runtime unchanged unless that runtime is Tested,
so the table cannot be contradicted by an older sentence elsewhere.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
STATUSES = {"Tested", "Coming"}
ISSUE = re.compile(r"^\[#(\d+)\]\(https://github\.com/philpaz/recusal/(?:issues|pull)/(\d+)\)$")
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

#: Runtimes no one has tested here yet. A sentence that names one of them next to a claim
#: that the gate fits it as-is would overclaim; such a runtime belongs in the table.
UNTESTED = ("OpenAI", "CrewAI", "AutoGen", "Google ADK", "Pydantic AI", "LlamaIndex", "Grok")
AS_IS_CLAIM = re.compile(r"drops into|unchanged|the gate is identical", re.IGNORECASE)


def _rows():
    block = README.split("<!-- works-with:start -->", 1)[1].split("<!-- works-with:end -->", 1)[0]
    lines = [ln for ln in block.strip().splitlines() if ln.startswith("|")]
    assert lines[0].startswith("| Runtime | Status | Where |"), lines[0]
    rows = []
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        assert len(cells) == 3, line
        rows.append(tuple(cells))
    return rows


def _slug(heading: str) -> str:
    """GitHub's anchor for a heading: lowercase, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def _anchors():
    return {_slug(h) for h in re.findall(r"^#{1,6} (.+)$", README, re.M)}


def test_table_parses_with_both_statuses():
    # guards the parser: an empty or partial parse must not pass by accident
    rows = _rows()
    assert len(rows) >= 8
    assert {status for _, status, _ in rows} == STATUSES


def test_every_status_is_known():
    for runtime, status, _ in _rows():
        assert status in STATUSES, f"{runtime}: unknown status {status!r}"


def test_tested_rows_link_to_proof_that_exists():
    tests_text = "\n".join(
        p.read_text(encoding="utf-8") for p in (ROOT / "tests").glob("test_*.py")
    )
    anchors = _anchors()
    for runtime, status, where in _rows():
        if status != "Tested":
            continue
        targets = LINK.findall(where)
        assert targets, f"{runtime}: a Tested row must link its proof"
        for target in targets:
            if target.startswith("#"):
                assert target[1:] in anchors, f"{runtime}: no README section {target}"
            else:
                assert target.startswith("examples/"), f"{runtime}: {target} is not an example"
                assert (ROOT / target).is_file(), f"{runtime}: {target} does not exist"
                stem = Path(target).stem
                assert stem in tests_text, f"{runtime}: no test exercises {target}"


def test_coming_rows_link_one_issue_or_pull_request():
    for runtime, status, where in _rows():
        if status != "Coming":
            continue
        match = ISSUE.match(where)
        assert match, f"{runtime}: a Coming row must link exactly one issue or PR, got {where!r}"
        assert match.group(1) == match.group(2), f"{runtime}: link text and URL disagree"


def _sentences(text: str):
    flat = re.sub(r"\s+", " ", text)
    return re.split(r"(?<=[.;:])\s", flat)


def test_no_document_claims_an_untested_runtime_works_as_is():
    documents = [
        ROOT / "README.md",
        *(ROOT / "docs").glob("*.md"),
        *(ROOT / "examples").glob("*.py"),
    ]
    offenders = []
    for path in documents:
        for sentence in _sentences(path.read_text(encoding="utf-8")):
            if AS_IS_CLAIM.search(sentence) and any(name in sentence for name in UNTESTED):
                offenders.append(f"{path.relative_to(ROOT)}: {sentence[:160]}")
    assert not offenders, "\n".join(offenders)


def test_the_overclaim_lock_catches_the_sentence_it_replaced():
    # negative control: the sentence this table replaced must still be refused
    old = (
        "the same `compute_verdict` seam drops into LangGraph, the OpenAI Agents SDK, or a "
        "homegrown runtime unchanged."
    )
    assert any(AS_IS_CLAIM.search(s) and any(n in s for n in UNTESTED) for s in _sentences(old))
