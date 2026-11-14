"""Regenerate ``assets/gate-position.svg`` and ``assets/gate-position-light.svg``, the
README's picture of where the gate sits: between the call an agent proposes and the tool
that would run it.

    python tools/render_gate.py            # rewrite both files
    python tools/render_gate.py --check    # exit 1 if either file is out of date

The three outcomes drawn are the three decisions the tool-call evaluator returns, and they
are asked of it at render time: ``decide`` is called with a policy that raises no objection,
one that fails, and one that raises, and the picture is refused if the answers are not
"the call goes ahead", "deny" and "deny". ``tests/test_gate_position.py`` fails if the
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

from recusal import Finding, decide  # noqa: E402

ASSETS = ROOT / "assets"
WIDTH, HEIGHT = 1200, 600
THEMES = {
    "gate-position.svg": _MAP_THEMES["adapter-map.svg"],
    "gate-position-light.svg": _MAP_THEMES["adapter-map-light.svg"],
}


def outcomes() -> dict[str, str]:
    """What the evaluator answers for a policy with no objection, one that fails, and one
    that raises. The picture's three claims, taken from the kernel."""

    def raises(_name, _input):
        raise RuntimeError("policy bug")

    def fails(_name, _input):
        return [Finding.fail("no", severity="CRITICAL", message="not allowed")]

    return {
        "no objection": decide("tool", {}, lambda _name, _input: [])[0],
        "a failing finding": decide("tool", {}, fails)[0],
        "a policy that raises": decide("tool", {}, raises, safe_errors=True)[0],
    }


def render(theme: dict[str, str], found: dict[str, str]) -> str:
    if found["a failing finding"] != "deny" or found["a policy that raises"] != "deny":
        raise SystemExit("the kernel no longer denies what this picture says it denies")
    if found["no objection"] == "deny":
        raise SystemExit("the kernel denies a call no policy objected to")
    t = theme
    box_w, box_h, top = 280, 96, 190
    xs = (70, 460, 850)
    mid = top + box_h // 2
    gate_x, gate_w = xs[1], box_w
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-labelledby="title desc">',
        '<title id="title">recusal: the gate sits between the proposed call and the tool</title>',
        '<desc id="desc">An agent proposes a tool call. The gate, your policy with no model in '
        "it, decides before the tool runs. With no objection the tool runs. On a failing "
        "finding, or when the policy itself raises, the call is refused: the tool never runs "
        "and the reason goes back to the model. Every decision can be written to a "
        "hash-chained audit log.</desc>",
        "<defs>",
        f'<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" '
        f'fill="{t["line"]}"/></marker>',
        f'<marker id="refuse" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" '
        f'fill="{t["accent"]}"/></marker>',
        "</defs>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{t["background"]}"/>',
        f'<rect x="22" y="22" width="{WIDTH - 44}" height="{HEIGHT - 44}" fill="none" '
        f'stroke="{t["frame"]}" stroke-width="1.5"/>',
        f'<text x="70" y="92" font-family="{MONO}" font-size="19" letter-spacing="6" '
        f'fill="{t["accent"]}">BEFORE THE TOOL RUNS</text>',
        f'<text x="{WIDTH - 70}" y="92" text-anchor="end" font-family="{MONO}" font-size="15" '
        f'letter-spacing="3" fill="{t["muted"]}">THE BUILDER DOES NOT GRADE ITS OWN WORK</text>',
    ]
    boxes = (
        (xs[0], "the agent", "proposes a tool call", False),
        (xs[1], "recusal", "your policy decides", True),
        (xs[2], "the tool", "runs only if not refused", False),
    )
    for x, title, line, is_gate in boxes:
        out.append(
            f'<rect x="{x}" y="{top}" width="{box_w}" height="{box_h}" rx="8" fill="{t["box"]}" '
            f'stroke="{t["accent"] if is_gate else t["line"]}" '
            f'stroke-width="{2 if is_gate else 1.2}"/>'
        )
        if is_gate:
            out.append(
                f'<text x="{x + box_w // 2}" y="{top + 50}" text-anchor="middle" '
                f'font-family="{SERIF}" font-size="40" font-weight="bold" fill="{t["ink"]}">'
                f'{title}<tspan fill="{t["accent"]}">.</tspan></text>'
            )
        else:
            out.append(
                f'<text x="{x + box_w // 2}" y="{top + 44}" text-anchor="middle" '
                f'font-family="{MONO}" font-size="20" fill="{t["ink"]}">{title}</text>'
            )
        out.append(
            f'<text x="{x + box_w // 2}" y="{top + 76}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="13" fill="{t["muted"]}">{line}</text>'
        )
    out += [
        # proposed call into the gate, and the allowed call on to the tool
        f'<path d="M{xs[0] + box_w} {mid}H{gate_x - 4}" fill="none" stroke="{t["line"]}" '
        f'stroke-width="1.6" marker-end="url(#arrow)"/>',
        f'<path d="M{gate_x + gate_w} {mid}H{xs[2] - 4}" fill="none" stroke="{t["line"]}" '
        f'stroke-width="1.6" marker-end="url(#arrow)"/>',
        f'<text x="{(gate_x + gate_w + xs[2]) // 2}" y="{mid - 12}" text-anchor="middle" '
        f'font-family="{MONO}" font-size="11" letter-spacing="1" fill="{t["muted"]}">'
        "NO OBJECTION</text>",
        f'<text x="{(xs[0] + box_w + gate_x) // 2}" y="{mid - 12}" text-anchor="middle" '
        f'font-family="{MONO}" font-size="11" letter-spacing="1" fill="{t["muted"]}">'
        "THE CALL</text>",
        # the refusal goes back to the agent, under the row, and never reaches the tool
        f'<path d="M{gate_x + 70} {top + box_h}V{top + box_h + 64}H{xs[0] + box_w // 2}'
        f'V{top + box_h + 4}" fill="none" stroke="{t["accent"]}" stroke-width="1.8" '
        f'marker-end="url(#refuse)"/>',
        f'<text x="{(xs[0] + box_w // 2 + gate_x + 70) // 2}" y="{top + box_h + 88}" '
        f'text-anchor="middle" font-family="{MONO}" font-size="13" letter-spacing="2" '
        f'fill="{t["accent"]}">REFUSED: THE REASON GOES BACK, THE TOOL NEVER RUNS</text>',
        # every decision can go on the record
        f'<path d="M{gate_x + gate_w - 70} {top + box_h}V{top + box_h + 120}" fill="none" '
        f'stroke="{t["line"]}" stroke-width="1.6" stroke-dasharray="6 5" '
        f'marker-end="url(#arrow)"/>',
        f'<rect x="{gate_x + gate_w - 70 - 190}" y="{top + box_h + 124}" width="380" height="62" '
        f'rx="6" fill="{t["box"]}" stroke="{t["line"]}" stroke-width="1.2"/>',
        f'<text x="{gate_x + gate_w - 70}" y="{top + box_h + 150}" text-anchor="middle" '
        f'font-family="{MONO}" font-size="16" fill="{t["ink"]}">audit log</text>',
        f'<text x="{gate_x + gate_w - 70}" y="{top + box_h + 172}" text-anchor="middle" '
        f'font-family="{MONO}" font-size="13" fill="{t["muted"]}">'
        "every decision, hash-chained, if you pass a log</text>",
        f'<text x="{xs[2] + box_w // 2}" y="{top + box_h + 150}" text-anchor="middle" '
        f'font-family="{MONO}" font-size="13" letter-spacing="1" fill="{t["muted"]}">'
        "A POLICY THAT RAISES</text>",
        f'<text x="{xs[2] + box_w // 2}" y="{top + box_h + 172}" text-anchor="middle" '
        f'font-family="{MONO}" font-size="13" letter-spacing="1" fill="{t["muted"]}">'
        "REFUSES THE CALL</text>",
        f'<text x="{WIDTH // 2}" y="{HEIGHT - 46}" text-anchor="middle" font-family="{SERIF}" '
        f'font-size="24" font-style="italic" fill="{t["muted"]}">'
        "A gate that cannot say no is not a gate.</text>",
        "</svg>",
    ]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if a file is out of date")
    args = parser.parse_args(argv)
    found = outcomes()
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
        print(f"out of date: {', '.join(stale)}; run tools/render_gate.py", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
