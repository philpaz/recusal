"""The README's picture of where the gate sits says what the tool-call evaluator does.

The picture makes three claims about one call: with no objection the tool runs, a failing
finding refuses it, and a policy that raises refuses it too. ``tools/render_gate.py`` asks
``decide`` for each before drawing, and these tests ask it again without the generator, so
the picture, the generator and the kernel have to agree.
"""

import importlib.util
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from recusal import Finding, decide
from recusal.audit import AuditLog

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
FILES = ("gate-position.svg", "gate-position-light.svg")
SVG = "{http://www.w3.org/2000/svg}"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("render_gate", ROOT / "tools" / "render_gate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fails(_name, _input):
    return [Finding.fail("no", severity="CRITICAL", message="not allowed")]


def _raises(_name, _input):
    raise RuntimeError("policy bug")


def test_the_kernel_does_what_the_picture_says():
    decision, _ = decide("tool", {}, lambda _name, _input: [])
    assert decision != "deny", "no objection: the tool runs"
    decision, reason = decide("tool", {}, _fails)
    assert decision == "deny" and "not allowed" in reason, "the reason goes back"
    decision, reason = decide("tool", {}, _raises, safe_errors=True)
    assert decision == "deny", "a policy that raises refuses the call"
    assert "policy bug" not in reason, "and its text is not handed to the model"


def test_every_decision_is_on_the_log_when_one_is_passed_and_nothing_is_written_without_one(
    tmp_path,
):
    log = AuditLog(path=str(tmp_path / "audit.jsonl"))
    for policy in (lambda _name, _input: [], _fails, _raises):
        decide("tool", {"a": 1}, policy, audit=log, surface="test.gate_position", safe_errors=True)
    assert len(log.entries) == 3, "allowed, refused and failed-closed are each one entry"
    assert [e["action"]["decision"] == "deny" for e in log.entries] == [False, True, True]
    before = sorted(p.name for p in tmp_path.iterdir())
    decide("tool", {"a": 1}, _fails)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_the_generator_asks_the_kernel_and_refuses_to_draw_a_false_picture(tool):
    found = tool.outcomes()
    assert found["a failing finding"] == "deny" and found["a policy that raises"] == "deny"
    assert found["no objection"] != "deny"
    theme = tool.THEMES[FILES[0]]
    with pytest.raises(SystemExit, match="no longer denies"):
        tool.render(theme, {**found, "a policy that raises": "allow"})
    with pytest.raises(SystemExit, match="denies a call no policy objected to"):
        tool.render(theme, {**found, "no objection": "deny"})


@pytest.mark.parametrize("name", FILES)
def test_the_committed_file_is_exactly_what_the_generator_prints(tool, name):
    assert (ASSETS / name).read_text(encoding="utf-8") == tool.render(
        tool.THEMES[name], tool.outcomes()
    ), f"assets/{name} is out of date; run python tools/render_gate.py"


def test_the_check_mode_passes_on_the_committed_files_and_fails_on_a_stale_one(
    tool, monkeypatch, tmp_path
):
    assert tool.main(["--check"]) == 0
    for name in FILES:
        (tmp_path / name).write_text((ASSETS / name).read_text(encoding="utf-8"), "utf-8")
    stale = tmp_path / FILES[0]
    stale.write_text(
        stale.read_text(encoding="utf-8").replace("THE TOOL NEVER RUNS", "THE TOOL RUNS"),
        encoding="utf-8",
    )
    monkeypatch.setattr(tool, "ASSETS", tmp_path)
    assert tool.main(["--check"]) == 1


@pytest.mark.parametrize("name", FILES)
def test_the_drawing_has_one_way_to_the_tool_and_one_way_back(name):
    root = ET.parse(ASSETS / name).getroot()
    texts = ["".join(node.itertext()) for node in root.iter(f"{SVG}text")]
    for wanted in (
        "the agent",
        "recusal.",
        "the tool",
        "NO OBJECTION",
        "REFUSED: THE REASON GOES BACK, THE TOOL NEVER RUNS",
        "audit log",
        "every decision, hash-chained, if you pass a log",
    ):
        assert wanted in texts, wanted
    paths = list(root.iter(f"{SVG}path"))
    refusals = [p for p in paths if p.get("marker-end") == "url(#refuse)"]
    forward = [p for p in paths if p.get("marker-end") == "url(#arrow)"]
    assert len(refusals) == 1, "one refusal path, back to the agent"
    assert len(forward) == 3, "the call in, the allowed call on, and the record"
    dashed = [p for p in forward if p.get("stroke-dasharray")]
    assert len(dashed) == 1, "the audit log is drawn as optional"
    # the line under the picture is the constitution's fourth rule
    line = "A gate that cannot say no is not a gate."
    assert line in texts
    assert "### 4. A gate that cannot say no is not a gate" in (ROOT / "CONSTITUTION.md").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("name", FILES)
def test_the_file_is_a_self_contained_accessible_image(name):
    text = (ASSETS / name).read_text(encoding="utf-8")
    root = ET.fromstring(text)
    assert root.get("role") == "img" and root.find(f"{SVG}title") is not None
    assert "the tool never runs" in root.find(f"{SVG}desc").text
    assert not list(root.iter(f"{SVG}script")) and not list(root.iter(f"{SVG}image"))
    assert "href" not in text and "<style" not in text and text.count("http") == 1
    assert "—" not in text


def test_the_readme_shows_both_variants_where_the_idea_is_introduced():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("## What Recusal does\n", 1)[1].split("\n## ", 1)[0]
    raw = "https://raw.githubusercontent.com/philpaz/recusal/main/assets/"
    assert f'srcset="{raw}gate-position.svg"' in section
    assert f'src="{raw}gate-position-light.svg"' in section
    alt = re.search(r'<img alt="([^"]+)"', section).group(1)
    for phrase in ("no objection the tool runs", "policy itself raises", "the tool never runs"):
        assert phrase in alt, phrase
    assert section.index("gate-position") < section.index("Recusal places deterministic policy")
