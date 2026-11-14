"""The README's picture of how a tool call becomes a verdict says what the kernel does.

``tools/render_pipeline.py`` asks ``compute_verdict`` which severity gives which verdict and
draws the answer. These tests hold the committed files to that output, hold the picture's
three cards to the kernel independently of the generator, and hold the README's words about
the picture to both.
"""

import importlib.util
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from recusal import Decision, Finding, Severity, compute_verdict

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
FILES = ("verdict-pipeline.svg", "verdict-pipeline-light.svg")
SVG = "{http://www.w3.org/2000/svg}"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location(
        "render_pipeline", ROOT / "tools" / "render_pipeline.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _kernel() -> dict:
    """Each verdict and the severities that produce it, asked of the kernel directly."""
    found: dict = {decision.name: [] for decision in Decision}
    for severity in Severity:
        verdict = compute_verdict([Finding.fail("x", severity=severity.name, message="m")])
        found[verdict.decision.name].append(severity.name)
    return found


def test_the_kernel_maps_severities_to_verdicts_as_the_picture_will_say():
    assert _kernel() == {"PASS": ["WARNING", "INFO"], "RETRY": ["ERROR"], "FAIL": ["CRITICAL"]}
    critical = compute_verdict([Finding.fail("x", severity="CRITICAL", message="m")])
    assert critical.refused, "the Fail card says refused"
    assert not compute_verdict([Finding.fail("x", severity="ERROR", message="m")]).refused


def test_the_generator_reads_the_mapping_from_the_kernel(tool):
    assert tool.verdicts() == _kernel()
    assert set(tool.MEANING) == {decision.name for decision in Decision}


@pytest.mark.parametrize("name", FILES)
def test_the_committed_file_is_exactly_what_the_generator_prints(tool, name):
    assert (ASSETS / name).read_text(encoding="utf-8") == tool.render(
        tool.THEMES[name], tool.verdicts()
    ), f"assets/{name} is out of date; run python tools/render_pipeline.py"


def test_the_check_mode_passes_on_the_committed_files_and_fails_on_a_stale_one(
    tool, monkeypatch, tmp_path
):
    assert tool.main(["--check"]) == 0
    for name in FILES:
        (tmp_path / name).write_text((ASSETS / name).read_text(encoding="utf-8"), "utf-8")
    stale = tmp_path / FILES[0]
    stale.write_text(
        stale.read_text(encoding="utf-8").replace("WORST FINDING: CRITICAL", "WORST FINDING: INFO"),
        encoding="utf-8",
    )
    monkeypatch.setattr(tool, "ASSETS", tmp_path)
    assert tool.main(["--check"]) == 1


@pytest.mark.parametrize("name", FILES)
def test_each_card_names_the_severities_the_kernel_gives_that_verdict(name):
    root = ET.parse(ASSETS / name).getroot()
    texts = ["".join(node.itertext()) for node in root.iter(f"{SVG}text")]
    for verdict, severities in _kernel().items():
        title = verdict.capitalize() + ("." if verdict == "FAIL" else "")
        assert title in texts, title
        assert f"WORST FINDING: {' OR '.join(severities)}" in texts
    assert sum(text.startswith("WORST FINDING:") for text in texts) == len(Decision)
    # one arrow between each pair of steps, and one into each verdict card
    arrows = [n for n in root.iter(f"{SVG}path") if n.get("marker-end") == "url(#arrow)"]
    assert len(arrows) == 2 + len(Decision)
    # only the refusing verdict carries the accent border
    accents = [n for n in root.iter(f"{SVG}rect") if n.get("stroke-width") == "2"]
    assert len(accents) == 1


@pytest.mark.parametrize("name", FILES)
def test_the_file_is_a_self_contained_accessible_image(name):
    text = (ASSETS / name).read_text(encoding="utf-8")
    root = ET.fromstring(text)
    assert root.get("role") == "img"
    assert root.find(f"{SVG}title") is not None
    assert "No model is in the decision path." in root.find(f"{SVG}desc").text
    assert not list(root.iter(f"{SVG}script")) and not list(root.iter(f"{SVG}image"))
    assert "href" not in text and "<style" not in text and text.count("http") == 1
    assert "—" not in text


def test_the_readme_shows_both_variants_and_says_in_words_what_the_picture_shows():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("## How it works\n", 1)[1].split("\n## ", 1)[0]
    raw = "https://raw.githubusercontent.com/philpaz/recusal/main/assets/"
    assert f'srcset="{raw}verdict-pipeline.svg"' in section
    assert f'src="{raw}verdict-pipeline-light.svg"' in section
    alt = re.search(r'<img alt="([^"]+)"', section).group(1)
    for verdict, severities in _kernel().items():
        assert f"A worst finding of {' or '.join(severities)} gives {verdict}" in alt
    # the modules the paragraph under the picture names are importable
    for module in re.findall(r"`(recusal\.[a-z_]+)`", section.split("</p>", 1)[1].split("|", 1)[0]):
        assert importlib.util.find_spec(module) is not None, module


def test_the_readme_table_under_the_picture_agrees_with_the_kernel():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    rows = dict(
        re.findall(r"^\| (`[A-Z]+`[^|]*?) (?:failure|only) *\| \*\*`([A-Z]+)`\*\* \|", readme, re.M)
    )
    seen = {}
    for cell, verdict in rows.items():
        for severity in re.findall(r"`([A-Z]+)`", cell):
            seen.setdefault(verdict, []).append(severity)
    assert seen == _kernel()
