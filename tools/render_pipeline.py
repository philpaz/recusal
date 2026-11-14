"""Regenerate ``assets/verdict-pipeline.svg`` and ``assets/verdict-pipeline-light.svg``, the
README's picture of how a tool call becomes a verdict.

    python tools/render_pipeline.py            # rewrite both files
    python tools/render_pipeline.py --check    # exit 1 if either file is out of date

The three verdicts and the severity that produces each are not typed here. They are asked
of the kernel at render time: one failing finding of each severity is folded with
``compute_verdict`` and the picture shows what came back, so it cannot say CRITICAL means
FAIL after the kernel stops saying so. ``tests/test_verdict_pipeline.py`` fails if the
committed files differ from what this prints.

It shares its palette and type with ``tools/render_adapter_map.py``. Standard library and
the kernel only, no embedded fonts, the same bytes on every machine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from render_adapter_map import MONO, SERIF  # noqa: E402
from render_adapter_map import THEMES as _MAP_THEMES  # noqa: E402

from recusal import Decision, Finding, Severity, compute_verdict  # noqa: E402

ASSETS = ROOT / "assets"
WIDTH, HEIGHT = 1200, 650
THEMES = {
    "verdict-pipeline.svg": _MAP_THEMES["adapter-map.svg"],
    "verdict-pipeline-light.svg": _MAP_THEMES["adapter-map-light.svg"],
}

#: The three steps, left to right: a title and the one line under it.
STEPS = (
    ("a proposed tool call", "the tool name and its arguments"),
    ("checks", "your policy returns findings"),
    ("compute_verdict", "the worst finding decides"),
)
#: What each verdict means for the call, in the README's words for it.
MEANING = {
    "PASS": ("Proceed.", "Warnings are recorded."),
    "RETRY": ("Recoverable.", "Retry once, with the failures."),
    "FAIL": ("Terminal.", "Refused before the call runs."),
}
#: What reads the verdict.
CONSUMERS = (
    ("gate", "allow or refuse the call"),
    ("audit", "a hash-chained record"),
    ("classify", "route the failure"),
)


def verdicts() -> dict[str, list[str]]:
    """Each verdict and the severities whose single failure produces it, from the kernel."""
    found: dict[str, list[str]] = {decision.name: [] for decision in Decision}
    for severity in Severity:
        finding = Finding.fail("probe", severity=severity.name, message="probe")
        found[compute_verdict([finding]).decision.name].append(severity.name)
    return found


def render(theme: dict[str, str], found: dict[str, list[str]]) -> str:
    t = theme
    box_w, box_h, top = 300, 76, 150
    xs = (70, 450, 830)
    card_w, card_h, card_y = 340, 176, 318
    card_xs = (70, 430, 790)
    bus_y = 280
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-labelledby="title desc">',
        '<title id="title">recusal: evidence in, one verdict out</title>',
        '<desc id="desc">A proposed tool call goes to checks, which return findings, and '
        "compute_verdict folds them into one verdict. "
        + " ".join(
            f"A worst finding of {' or '.join(severities)} gives {name}."
            for name, severities in found.items()
        )
        + " The verdict drives the gate, the audit log and the failure classifier. "
        "No model is in the decision path.</desc>",
        "<defs>",
        f'<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" '
        f'fill="{t["line"]}"/></marker>',
        "</defs>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{t["background"]}"/>',
        f'<rect x="22" y="22" width="{WIDTH - 44}" height="{HEIGHT - 44}" fill="none" '
        f'stroke="{t["frame"]}" stroke-width="1.5"/>',
        f'<text x="70" y="92" font-family="{MONO}" font-size="19" letter-spacing="6" '
        f'fill="{t["accent"]}">EVIDENCE IN · ONE VERDICT OUT</text>',
        f'<text x="{WIDTH - 70}" y="92" text-anchor="end" font-family="{MONO}" font-size="15" '
        f'letter-spacing="3" fill="{t["muted"]}">NO MODEL IN THE DECISION PATH</text>',
    ]
    for index, (x, (title, line)) in enumerate(zip(xs, STEPS)):
        out += [
            f'<rect x="{x}" y="{top}" width="{box_w}" height="{box_h}" rx="6" '
            f'fill="{t["box"]}" stroke="{t["line"]}" stroke-width="1.2"/>',
            f'<text x="{x + box_w // 2}" y="{top + 32}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="18" fill="{t["ink"]}">{title}</text>',
            f'<text x="{x + box_w // 2}" y="{top + 57}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="13" fill="{t["muted"]}">{line}</text>',
        ]
        if index:
            out.append(
                f'<path d="M{xs[index - 1] + box_w} {top + box_h // 2}H{x - 4}" fill="none" '
                f'stroke="{t["line"]}" stroke-width="1.6" marker-end="url(#arrow)"/>'
            )
    # the verdict leaves compute_verdict and lands on exactly one of the three cards
    centre = xs[2] + box_w // 2
    out.append(
        f'<path d="M{centre} {top + box_h}V{bus_y}" fill="none" stroke="{t["line"]}" '
        f'stroke-width="1.6"/>'
    )
    out.append(
        f'<path d="M{card_xs[0] + card_w // 2} {bus_y}H{max(centre, card_xs[2] + card_w // 2)}" '
        f'fill="none" stroke="{t["line"]}" stroke-width="1.6"/>'
    )
    for x, (name, severities) in zip(card_xs, found.items()):
        mid = x + card_w // 2
        refuses = name == "FAIL"
        first, second = MEANING[name]
        out += [
            f'<path d="M{mid} {bus_y}V{card_y - 4}" fill="none" stroke="{t["line"]}" '
            f'stroke-width="1.6" marker-end="url(#arrow)"/>',
            f'<rect x="{x}" y="{card_y}" width="{card_w}" height="{card_h}" rx="8" '
            f'fill="{t["box"]}" stroke="{t["accent"] if refuses else t["line"]}" '
            f'stroke-width="{2 if refuses else 1.2}"/>',
            f'<text x="{x + 28}" y="{card_y + 66}" font-family="{SERIF}" font-size="46" '
            f'font-weight="bold" fill="{t["ink"]}">{name.capitalize()}'
            + (f'<tspan fill="{t["accent"]}">.</tspan>' if refuses else "")
            + "</text>",
            f'<text x="{x + 28}" y="{card_y + 100}" font-family="{MONO}" font-size="13" '
            f'letter-spacing="1" fill="{t["muted"]}">WORST FINDING: '
            f"{' OR '.join(severities)}</text>",
            f'<text x="{x + 28}" y="{card_y + 130}" font-family="{MONO}" font-size="15" '
            f'fill="{t["ink"]}">{first}</text>',
            f'<text x="{x + 28}" y="{card_y + 152}" font-family="{MONO}" font-size="15" '
            f'fill="{t["ink"]}">{second}</text>',
        ]
    consumer_y = card_y + card_h + 52
    out.append(
        f'<text x="70" y="{consumer_y}" font-family="{MONO}" font-size="14" letter-spacing="3" '
        f'fill="{t["accent"]}">THE VERDICT DRIVES</text>'
    )
    for x, (name, does) in zip((330, 620, 900), CONSUMERS):
        out.append(
            f'<text x="{x}" y="{consumer_y}" font-family="{MONO}" font-size="15" '
            f'fill="{t["ink"]}">{name}<tspan fill="{t["muted"]}">  {does}</tspan></text>'
        )
    out += [
        f'<text x="{WIDTH // 2}" y="{HEIGHT - 50}" text-anchor="middle" font-family="{SERIF}" '
        f'font-size="24" font-style="italic" fill="{t["muted"]}">'
        "Same evidence, same verdict.</text>",
        "</svg>",
    ]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if a file is out of date")
    args = parser.parse_args(argv)
    found = verdicts()
    stale = []
    for name, theme in THEMES.items():
        text = render(theme, found)
        path = ASSETS / name
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(name)
        else:
            path.write_bytes(text.encode("utf-8"))
            print(f"wrote assets/{name}")
    if stale:
        print(f"out of date: {', '.join(stale)}; run tools/render_pipeline.py", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
